#!/usr/bin/env python3
from argparse import ArgumentParser
from collections import Counter
import gzip
import json
from pathlib import Path
import pickle

from data_path_utils import (
    create_data_path,
    create_output_path,
    find_newest_data_path,
)
import pandas as pd

from gene_mappings import read_ensembl_entrez_mapping
from utils import sorted_union

p = ArgumentParser()
p.add_argument('gdc_manifest', type=Path)
args = p.parse_args()

data_path = create_data_path('consolidate_mrna_expression')

input_path = find_newest_data_path('query_cases_by_file') / 'raw_responses'

gdc_manifest = pd.read_table(args.gdc_manifest)
files_in_manifest = set(gdc_manifest.id)

def get_submitter_ids(data: dict):
    for key, value in data.items():
        if key == 'submitter_id':
            yield value
        if isinstance(value, dict):
            yield from get_submitter_ids(value)
        if isinstance(value, list):
            for sub_data in value:
                yield from get_submitter_ids(sub_data)

json_data_by_file = {}
patient_id_by_file_id = {}
patients_with_files = set()

for filepath in input_path.iterdir():
    file_id = filepath.stem
    with filepath.open() as f:
        json_data = json.load(f)
    json_data_by_file[file_id] = json_data
    submitter_ids = set(get_submitter_ids(json_data))
    assert len(submitter_ids) == 1
    patient_id = next(iter(submitter_ids))
    patient_id_by_file_id[file_id] = patient_id
    patients_with_files.add(patient_id)

files_on_disk = set(json_data_by_file)
disk_but_not_manifest = files_on_disk - files_in_manifest
if disk_but_not_manifest:
    print('Files on disk but not in manifest:')
    for file_id in disk_but_not_manifest:
        print('\t{}'.format(file_id))
manifest_but_not_disk = files_in_manifest - files_on_disk
if manifest_but_not_disk:
    print('Files in manifest but not on disk:')
    for file_id in manifest_but_not_disk:
        print('\t{}'.format(file_id))

expr_data_path = Path('~/data/tcga-brca-mrna').expanduser()

EXPRESSION_PATTERN = '*.FPKM.txt.gz'
def find_mrna_expression_file(file_id: str) -> Path:
    patient_dir = expr_data_path / file_id
    expr_data_files = list(patient_dir.glob(EXPRESSION_PATTERN))
    assert len(expr_data_files) == 1
    return expr_data_files[0]

ensembl_entrez_mapping = read_ensembl_entrez_mapping()

def relabel_expr(expr: pd.DataFrame) -> pd.Series:
    """
    1. Select Ensembl IDs that are present in ensembl_entrez_mapping
    2. Create intermediate Series with only those genes
    3. Replace Series index with Entrez IDs
    4. Identify duplicate Entrez IDs
    5. Create new Series with unique Entrez IDs
    6. Copy expression values for unique entries
    7. Calculate median of each duplicated gene, assign

    TODO use Pandas groupby for this
    """

    # Separate lists for ease of indexing and replacing columns, though
    # elements in each list directly correspond. ensembl_ids_in_entrez
    # will be used to index into `expr`, then entrez_ids will be used
    # as a replacement index
    ensembl_ids_in_entrez = []
    entrez_ids = []
    for ensembl_id_full in expr.index:
        ensembl_id, *junk = ensembl_id_full.split('.')
        if ensembl_id in ensembl_entrez_mapping:
            ensembl_ids_in_entrez.append(ensembl_id_full)
            entrez_ids.append(ensembl_entrez_mapping[ensembl_id])

    subset = expr.loc[ensembl_ids_in_entrez, :].iloc[:, 0]
    subset.index = entrez_ids

    entrez_id_counts = Counter(entrez_ids)
    single_genes = {gene for (gene, count) in entrez_id_counts.items() if count == 1}
    duplicated_genes = {gene for (gene, count) in entrez_id_counts.items() if count > 1}

    s = pd.Series(index=entrez_id_counts)
    s.loc[single_genes] = subset.loc[single_genes]
    for gene in duplicated_genes:
        s.loc[gene] = subset.loc[gene].median()

    return s

expr_data_by_patient = {}

for file_id, patient_id in patient_id_by_file_id.items():
    expr_file = find_mrna_expression_file(file_id)
    print('Reading', expr_file)
    with gzip.open(expr_file, 'rt') as f:
        expr_temp = pd.read_table(f, header=None, index_col=0)
        expr_data_by_patient[patient_id] = relabel_expr(expr_temp)

expr_by_patient_output_path = data_path / 'mrna_expr_raw_by_patient.pickle'
print('Saving raw mRNA expression by patient to', expr_by_patient_output_path)
with open(expr_by_patient_output_path, 'wb') as f:
    pickle.dump(expr_data_by_patient, f)

patients = sorted(expr_data_by_patient)
all_mrnas = sorted_union(*(e.index for e in expr_data_by_patient.values()))

expr_data = pd.DataFrame(dtype=float, index=patients, columns=all_mrnas)
for patient_id, e in expr_data_by_patient.items():
    expr_data.loc[patient_id, e.index] = e.as_matrix()

expr_output_path = data_path / 'mrna_expr.pickle'
print('Saving mRNA expression to', expr_output_path)
expr_data.to_pickle(expr_output_path)
