#!/usr/bin/env python3
from argparse import ArgumentParser

from data_path_utils import (
    create_data_path,
    create_output_path,
    find_newest_data_path,
)
import numpy as np
import pandas as pd
from scipy.special import expit

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

script_label = 'pmid_26892682_analysis'
data_path = create_data_path(script_label)
output_path = create_output_path(script_label)

genes = ['TWIST1', 'KRT81', 'PTRF', 'EEF1A2', 'PTPRK', 'EGFR', 'CXCL14', 'ERBB3']
t_value_strs = ['-2.879', '-2.453', '-2.024', '-1.895', '-1.793', '-1.701', '2.229', '2.26']
t_values_inverted = np.array([float(v) for v in t_value_strs])
t_values = -t_values_inverted
coefs_all = pd.Series(t_values, index=genes)

expr_path = find_newest_data_path('parse_cosmic_diffexpr')
expr = pd.read_pickle(expr_path / 'brca_expr.pickle')

selected_genes = sorted_intersection(coefs_all.index, expr.columns)
coefs = coefs_all.loc[selected_genes]

def pmid_26892682_analysis(drug: str):
    feature_label_path = find_newest_data_path(f'compute_drug_features_labels_alpha_{args.alpha:.2f}')
    labels_all = pd.read_pickle(feature_label_path / f'labels_{drug}.pickle')

    selected_samples = sorted_intersection(labels_all.index, expr.index)
    selected_expr = expr.loc[selected_samples, selected_genes]
    selected_labels = labels_all.loc[selected_samples]

    ln_p_over_1_minus_p = selected_expr.as_matrix() @ coefs.as_matrix()
    probs = expit(ln_p_over_1_minus_p)

    rd = RocData.calculate(selected_labels, probs)
    rd.save(data_path / f'roc_data_{drug}.pickle')
    plot_roc(rd, f'PMID26892682 ROC: {drug.title()}', output_path / f'{drug}_roc.pdf')

    pr = PrData.calculate(selected_labels, probs)
    plot_pr(pr, f'PMID26892682 Precision-Recall: {drug.title()}', output_path / f'{drug}_pr.pdf')

if __name__ == '__main__':
    pmid_26892682_analysis('ai_all')
    pmid_26892682_analysis('arimidex')
