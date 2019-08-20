#!/usr/bin/env python3
from data_path_utils import (
    DATA_PATH,
    create_data_path,
    find_newest_data_path,
)
import pandas as pd
from scipy.stats import pearsonr

from utils import consolidate_data_frames, sorted_intersection

data_path = create_data_path('tcga_lincs_expr_features')

drugs = [
    'arimidex',
    'taxol',
]

tcga_expr_path = find_newest_data_path('parse_cosmic_diffexpr') / 'brca_expr.pickle'
print('Reading expression data from', tcga_expr_path)
tcga_expr = pd.read_pickle(tcga_expr_path)

lincs_expr = pd.read_csv(
    find_newest_data_path('gct_drug_subset') / 'subset.csv',
    header=None,
    index_col=0,
)
lincs_expr.columns = drugs

lincs_genes = set(lincs_expr.index)
tcga_genes = set(tcga_expr.columns)

lincs_benchmark_gene_data = pd.read_excel(DATA_PATH / 'Landmark_Genes_n978.xlsx')
lincs_benchmark_genes = set(lincs_benchmark_gene_data.loc[:, 'Gene Symbol'])

common_genes = sorted_intersection(lincs_genes, tcga_genes, lincs_benchmark_genes)
tcga_only_genes = tcga_genes - lincs_genes
lincs_only_genes = lincs_benchmark_genes - tcga_genes

print('Intersection of TCGA and LINCS gene symbols: {} genes'.format(len(common_genes)))
print('Gene symbols only in TCGA expression data: {}'.format(len(tcga_only_genes)))
print('Gene symbols only in LINCS expression data: {}'.format(len(lincs_only_genes)))

lincs_expr_common_with_dups = lincs_expr.loc[common_genes, :].fillna(0)
lincs_expr_common = pd.DataFrame(0.0, index=common_genes, columns=drugs)
for i, rows in lincs_expr_common_with_dups.groupby(lincs_expr_common_with_dups.index):
    lincs_expr_common.loc[i, :] = rows.mean()

tcga_expr_common = tcga_expr.loc[:, common_genes].fillna(0)

lincs_corr = pd.DataFrame(0.0, index=tcga_expr.index, columns=drugs)
lincs_dot_product = pd.DataFrame(0.0, index=tcga_expr.index, columns=drugs)

for drug in drugs:
    drug_expr = lincs_expr_common.loc[:, drug]
    for sample in tcga_expr.index:
        sample_expr = tcga_expr_common.loc[sample, :]
        corr, pv = pearsonr(drug_expr, sample_expr)
        lincs_corr.loc[sample, drug] = corr

        lincs_dot_product.loc[sample, drug] = drug_expr.as_matrix() @ sample_expr.as_matrix()

both_data = consolidate_data_frames([(lincs_corr, 'corr'), (lincs_dot_product, 'dot_product')])

data_output_path = data_path / 'data.pickle'
print('Saving LINCS/TCGA expression features to', data_output_path)
both_data.to_pickle(data_output_path)
