#!/usr/bin/env python3
from argparse import ArgumentParser
import gzip
from pathlib import Path

from data_path_utils import create_data_path
import pandas as pd
import scipy.sparse

from tcga import get_patient_barcode

def parse_cosmic_diffexpr(cosmic_expr_path: Path, patient_id_file: Path):
    """
    :param cosmic_expr_path: Full COSMIC data file, in .tsv.gz format
    :param patient_id_file: File containing TCGA sample IDs, one per line, which will
        be selected from the entire COSMIC set
    :return:
    """
    with patient_id_file.open() as f:
        patient_ids = set(line.strip() for line in f)

    print('Selected {} patient IDs'.format(len(patient_ids)))

    def is_selected(sample_id: str) -> bool:
        return sample_id in patient_ids

    with gzip.open(cosmic_expr_path, mode='rt') as g:
        cosmic = pd.read_table(g)

    print('Read COSMIC gene expression data')

    cosmic.loc[:, 'adj_sample_name'] = cosmic.SAMPLE_NAME.apply(get_patient_barcode)
    selected_diffexpr = cosmic.loc[cosmic.adj_sample_name.apply(is_selected), :]

    expr_samples = sorted(set(selected_diffexpr.adj_sample_name))
    expr_genes = sorted(set(selected_diffexpr.GENE_NAME))

    sample_mapping = {sample: i for (i, sample) in enumerate(expr_samples)}
    gene_mapping = {gene: i for (i, gene) in enumerate(expr_genes)}

    sample_vec = [sample_mapping[sample] for sample in selected_diffexpr.adj_sample_name]
    gene_vec = [gene_mapping[gene] for gene in selected_diffexpr.GENE_NAME]

    mat = scipy.sparse.coo_matrix((selected_diffexpr.Z_SCORE, (sample_vec, gene_vec)))
    return pd.DataFrame(mat.todense(), index=expr_samples, columns=expr_genes)

if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('cosmic_filename', type=Path)
    p.add_argument('patient_id_file', type=Path)
    args = p.parse_args()
    diffexpr = parse_cosmic_diffexpr(args.cosmic_filename, args.patient_id_file)

    data_path = create_data_path('parse_cosmic_diffexpr')
    diffexpr.to_pickle(data_path / 'diffexpr.pickle')
