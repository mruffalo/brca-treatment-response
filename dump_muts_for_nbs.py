#!/usr/bin/env python3
from data_path_utils import (
    create_data_path,
    find_newest_data_path,
)
import pandas as pd

from gene_mappings import read_entrez_hugo_mapping, read_hugo_entrez_mapping

data_path = create_data_path('dump_muts_for_nbs')

hugo_entrez_mapping = read_hugo_entrez_mapping()
entrez_hugo_mapping = read_entrez_hugo_mapping()

mut_path = find_newest_data_path('parse_tcga_mutations')

muts_all = pd.read_pickle(mut_path / 'mutations.pickle')
gene_sel = pd.Series(
    [
        (gene in hugo_entrez_mapping and hugo_entrez_mapping[gene])
        for gene in muts_all.columns
    ],
    index=muts_all.columns,
).astype(bool)
muts = muts_all.loc[:, gene_sel]
muts.columns = [hugo_entrez_mapping[gene] for gene in muts.columns]
muts = muts.groupby(axis=1, level=-1).any().astype(int)
gene_symbols = [entrez_hugo_mapping[gene] for gene in muts.columns]

with open(data_path / 'gene_symbols.txt', 'w') as f:
    print('Gene', file=f)
    for gene_symbol in gene_symbols:
        print(gene_symbol, file=f)

muts.to_csv(data_path / 'muts.csv', header=False, index=False)

with open(data_path / 'gene_ids.txt', 'w') as f:
    for gene in muts.columns:
        print(gene, file=f)

with open(data_path / 'patient_ids.txt', 'w') as f:
    print('Patient', file=f)
    for patient_id in muts.index:
        print(patient_id, file=f)
