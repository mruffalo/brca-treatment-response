#!/usr/bin/env python3
from argparse import ArgumentParser
from math import nan
from typing import Union

from data_path_utils import (
    create_data_path,
    create_output_path,
    find_newest_data_path,
)
import numpy as np
import pandas as pd

from utils import (
    DEFAULT_ALPHA,
    PrData,
    RocData,
    plot_pr,
    plot_roc,
    sorted_intersection,
)

drugs = ['ai_all', 'arimidex']

Child = Union['TreeNode', int]

class TreeNode:
    gene: str
    low: Child
    high: Child
    threshold: float

    def __init__(self, gene: str, low: Child, high: Child):
        self.gene = gene
        self.low = low
        self.high = high
        self.threshold = nan

    def __repr__(self):
        return f'<{type(self).__qualname__}: gene {self.gene}, threshold {self.threshold}>'

def build_tree():
    tree = TreeNode(
        'IL6ST',
        low=TreeNode(
            'ASPM',
            low=TreeNode(
                'NGFRAP1',
                low=1,
                high=0,
            ),
            high=1,
        ),
        high=TreeNode(
            'MCM4',
            low=0,
            high=TreeNode(
                'ASPM',
                low=0,
                high=TreeNode(
                    'NGFRAP1',
                    low=1,
                    high=0,
                )
            )
        )
    )
    return tree

p = ArgumentParser()
p.add_argument('--alpha', type=float, default=DEFAULT_ALPHA)
if '__file__' in globals():
    args = p.parse_args()
else:
    args = p.parse_args([])

script_label = 'pmid_26033813_analysis'
data_path = create_data_path(script_label)
output_path = create_output_path(script_label)

expr_path = find_newest_data_path('parse_cosmic_diffexpr')
expr = pd.read_pickle(expr_path / 'brca_expr.pickle')

def gini_impurity(labels_low: pd.Series, labels_high: pd.Series) -> float:
    return 1 - labels_low.mean() ** 2 - labels_high.mean() ** 2

def fit_tree(expr_data, labels: pd.Series, root: Child):
    if isinstance(root, int):
        return

    gene = root.gene
    expr = expr_data.loc[:, gene]

    sorted_expr = expr.sort_values().as_matrix()
    thresholds = np.unique((sorted_expr[1:] + sorted_expr[:-1]) / 2)

    if not thresholds.shape[0]:
        # Can't assign; not enough data left
        return

    gini_impurities = pd.Series(nan, index=thresholds)
    for threshold in thresholds:
        samples_high = expr.index[expr >= threshold]
        labels_high = labels.loc[samples_high]
        samples_low = expr.index[expr < threshold]
        labels_low = labels.loc[samples_low]

        gini_impurities.loc[threshold] = gini_impurity(labels_low, labels_high)

    best_threshold = gini_impurities.argmin()

    root.threshold = best_threshold

    samples_high = expr.index[expr >= best_threshold]
    samples_low = expr.index[expr < best_threshold]

    fit_tree(expr_data.loc[samples_high, :], labels.loc[samples_high], root.high)
    fit_tree(expr_data.loc[samples_low, :], labels.loc[samples_low], root.low)

def predict_sample(sample_name, expr_data, root: Child) -> int:
    if isinstance(root, int):
        return root

    sample_gene_expr = expr_data.loc[sample_name, root.gene]
    if sample_gene_expr >= root.threshold:
        return predict_sample(sample_name, expr_data, root.high)
    else:
        return predict_sample(sample_name, expr_data, root.low)

def pmid_26033813_analysis(drug: str):
    tree = build_tree()

    feature_label_path = find_newest_data_path(f'compute_drug_features_labels_alpha_{args.alpha:.2f}')
    labels_all = pd.read_pickle(feature_label_path / f'labels_{drug}.pickle')

    selected_samples = sorted_intersection(labels_all.index, expr.index)
    selected_labels = labels_all.loc[selected_samples]
    selected_expr = expr.loc[selected_samples, :]

    fit_tree(selected_expr, selected_labels, tree)

    predictions = pd.Series(
        [predict_sample(sample_name, selected_expr, tree) for sample_name in selected_samples],
        index=selected_samples,
    )

    rd = RocData.calculate(selected_labels, predictions)
    rd.save(data_path / f'roc_data_{drug}.pickle')
    plot_roc(rd, f'PMID26033813 ROC: {drug.title()}', output_path / f'{drug}_roc.pdf')

    pr = PrData.calculate(selected_labels, predictions)
    plot_pr(pr, f'PMID26033813 Precision-Recall: {drug.title()}', output_path / f'{drug}_pr.pdf')

if __name__ == '__main__':
    pmid_26033813_analysis('ai_all')
    pmid_26033813_analysis('arimidex')
