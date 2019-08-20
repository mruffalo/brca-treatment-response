#!/usr/bin/env python3
from argparse import ArgumentParser
from itertools import permutations
import os
from typing import List

from data_path_utils import (
    create_data_path,
    create_output_path,
    find_newest_data_path,
)
import matplotlib

if '__file__' in globals() or 'SSH_CONNECTION' in os.environ:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io

from utils import (
    CROSSVAL_FIGSIZE,
    DEFAULT_ALPHA,
    RocData,
    SIGNIFICANT_DIGITS,
    new_plot,
    plot_roc,
    sorted_intersection,
)

p = ArgumentParser()
p.add_argument('--alpha', type=float, default=DEFAULT_ALPHA)

if '__file__' in globals():
    args = p.parse_args()
else:
    args = p.parse_args([])

script_label = 'snf_cluster_results'
data_path = create_data_path(script_label)
output_path = create_output_path(script_label)

snf_path = find_newest_data_path('run_snf')
snf_data = pd.read_csv(snf_path / 'clustering.csv', index_col=0)
snf_data.columns = ['cluster']
cluster_assignments = snf_data.iloc[:, 0]

clusters = sorted(set(cluster_assignments))

def get_cluster_permutation(permutation: List[int]) -> pd.Series:
    mapping = dict(zip(clusters, permutation))
    relabeled = pd.Series(
        [mapping[value] for value in cluster_assignments],
        index=cluster_assignments.index,
    )
    return relabeled

possible_orders = permutations(clusters)
reordered_labels = []
for order in possible_orders:
    reordered_labels.append((order, get_cluster_permutation(order)))

drugs = ['ai_all', 'arimidex']

feature_label_path = find_newest_data_path(f'compute_drug_features_labels_alpha_{args.alpha:.2f}')

aucs = pd.Series(0, index=range(len(reordered_labels)))

for drug in drugs:
    roc_data = []

    for i, (order, clusters) in enumerate(reordered_labels):
        labels_all = pd.read_pickle(feature_label_path / f'labels_{drug}.pickle')

        selected_samples = sorted_intersection(labels_all.index, clusters.index)
        selected_labels = labels_all.loc[selected_samples]
        selected_clusters = clusters.loc[selected_samples]

        rd = RocData.calculate(selected_labels, selected_clusters)
        rd.save(data_path / f'roc_data_{drug}_permutation_{i}.pickle')
        roc_data.append(rd)

        aucs.loc[i] = rd.auc

    with new_plot():
        plt.figure(figsize=CROSSVAL_FIGSIZE)
        for i, rd in enumerate(roc_data):
            plt.plot(rd.fpr, rd.tpr, lw=1, label=f'Permutation {i} (area = {rd.auc:.{SIGNIFICANT_DIGITS}f})')

        plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

        plt.xlim([-0.01, 1.01])
        plt.ylim([-0.01, 1.01])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.axes().set_aspect('equal', 'datalim')
        plt.legend(loc='lower right')
        plt.title(f'SNF Clustering: {drug.title()}')

        figure_path = output_path / f'roc_comparison_{drug}.pdf'
        print('Saving ROC plot to', figure_path)
        plt.savefig(str(figure_path), bbox_inches='tight')

    best_permutation = aucs.idxmax()
    rd = roc_data[best_permutation]

    plot_roc(
        roc_data[best_permutation],
        f'SNF Clustering ROC: {drug.title()}',
        output_path / f'roc_best_{drug}.pdf',
    )

    rd.save(data_path / f'roc_data_{drug}.pickle')
