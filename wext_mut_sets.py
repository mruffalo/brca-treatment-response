#!/usr/bin/env python3
from argparse import ArgumentParser

from data_path_utils import (
    create_data_path,
    create_output_path,
    find_newest_data_path,
)
import pandas as pd

from gene_mappings import read_hugo_entrez_mapping
from utils import (
    DEFAULT_ALPHA,
    RocData,
    plot_roc,
    sorted_intersection,
)

script_label = 'wext_mut_sets'
data_path = create_data_path(script_label)
output_path = create_output_path(script_label)

p = ArgumentParser()
p.add_argument('--alpha', type=float, default=DEFAULT_ALPHA)
if '__file__' in globals():
    args = p.parse_args()
else:
    args = p.parse_args([])

hugo_entrez_mapping = read_hugo_entrez_mapping()

# Manually created
input_path = find_newest_data_path('wext_results')

gene_set_data = pd.read_table(input_path / 'tcga-exclusive-sets-sampled-sets.tsv')

cols = list(gene_set_data.columns)
cols[:2] = ['gene_set', 'pvalue']
gene_set_data.columns = cols

cutoff = 0.002
selected_gene_set_strs = gene_set_data.loc[gene_set_data.pvalue < cutoff, 'gene_set']
selected_gene_sets = [set(gene_set.split(',')) for gene_set in selected_gene_set_strs]

entrez_gene_sets = [
    {hugo_entrez_mapping[gene] for gene in gene_set if gene in hugo_entrez_mapping}
    for gene_set in selected_gene_sets
]

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

patient_gene_set_muts = pd.DataFrame(0, index=muts.index, columns=range(len(entrez_gene_sets)))

for i, gene_set in enumerate(entrez_gene_sets):
    patient_gene_set_muts.loc[:, i] = muts.loc[:, gene_set].any(axis=1).astype(int)

pathway_mut_counts = patient_gene_set_muts.sum(axis=1)

gene_set_mut_matrix_path = data_path / 'gene_set_mut_matrix.pickle'
print('Saving gene set mutation matrix to', gene_set_mut_matrix_path)
patient_gene_set_muts.to_pickle(gene_set_mut_matrix_path)

pathway_mut_count_path = data_path / 'pathway_mut_counts.pickle'
print('Saving pathway mutation counts to', pathway_mut_count_path)
pathway_mut_counts.to_pickle(pathway_mut_count_path)

drugs = ['ai_all', 'arimidex']

feature_label_path = find_newest_data_path(f'compute_drug_features_labels_alpha_{args.alpha:.2f}')

for drug in drugs:
    labels_all = pd.read_pickle(feature_label_path / f'labels_{drug}.pickle')

    selected_samples = sorted_intersection(labels_all.index, pathway_mut_counts.index)
    selected_labels = labels_all.loc[selected_samples]
    selected_counts = pathway_mut_counts.loc[selected_samples]

    rd = RocData.calculate(selected_labels, selected_counts)
    rd.save(data_path / f'roc_data_{drug}.pickle')
    plot_roc(rd, f'WExT Pathway Mutation Count ROC: {drug.title()}', output_path / f'{drug}_roc.pdf')
