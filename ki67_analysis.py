#!/usr/bin/env python3
from argparse import ArgumentParser

from data_path_utils import (
    create_data_path,
    create_output_path,
    find_newest_data_path,
)
import pandas as pd

from utils import (
    DEFAULT_ALPHA,
    PrData,
    RocData,
    plot_pr,
    plot_roc,
    sorted_intersection,
)

p = ArgumentParser()
p.add_argument('--alpha', type=float, default=DEFAULT_ALPHA)
if '__file__' in globals():
    args = p.parse_args()
else:
    args = p.parse_args([])

script_label = 'ki67_analysis'
data_path = create_data_path(script_label)
output_path = create_output_path(script_label)

expr_path = find_newest_data_path('parse_cosmic_diffexpr')
expr = pd.read_pickle(expr_path / 'brca_expr.pickle')

gene = 'MKI67'

def ki67_analysis(drug: str):
    feature_label_path = find_newest_data_path(f'compute_drug_features_labels_alpha_{args.alpha:.2f}')
    labels_all = pd.read_pickle(feature_label_path / f'labels_{drug}.pickle')

    selected_samples = sorted_intersection(labels_all.index, expr.index)
    selected_expr = expr.loc[selected_samples, gene]
    selected_labels = labels_all.loc[selected_samples]

    rd = RocData.calculate(selected_labels, selected_expr)
    rd.save(data_path / f'roc_data_{drug}.pickle')
    plot_roc(rd, f'Ki67 ROC: {drug.title()}', output_path / f'{drug}_roc.pdf')

    pr = PrData.calculate(selected_labels, selected_expr)
    plot_pr(pr, f'Ki67 Precision-Recall: {drug.title()}', output_path / f'{drug}_pr.pdf')

if __name__ == '__main__':
    ki67_analysis('ai_all')
    ki67_analysis('arimidex')
