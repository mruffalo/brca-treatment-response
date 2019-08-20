#!/usr/bin/env python3
from argparse import ArgumentParser
import os
from pathlib import Path
import pickle
from typing import Dict, List, Tuple

from data_path_utils import (
    create_output_path,
    find_newest_data_path,
)
import matplotlib
if '__file__' in globals() or 'SSH_CONNECTION' in os.environ:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import (
    BASE_FPR,
    CROSSVAL_FIGSIZE,
    DEFAULT_ALPHA,
    RocData,
    SIGNIFICANT_DIGITS,
    drug_name_to_description,
    new_plot,
)

p = ArgumentParser()
p.add_argument('--alpha', type=float, default=DEFAULT_ALPHA)

if '__file__' in globals():
    args = p.parse_args()
else:
    args = p.parse_args([])

output_path = create_output_path('consolidated_roc_plot')

fold_count = 5

# [0] are labels, [1] are data paths
single_curve_input_paths = [
    (r'Turnbull $\it{et}$ $\it{al.}$', find_newest_data_path('pmid_26033813_analysis')),
    (r'Reijm $\it{et}$ $\it{al.}$', find_newest_data_path('pmid_26892682_analysis')),
    (r'WExT Mutation Set Count', find_newest_data_path('wext_mut_sets')),
    (r'Network Based Stratification', find_newest_data_path('nbs_cluster_results')),
    (r'Similarity Network Fusion', find_newest_data_path('snf_cluster_results')),
]

crossval_input_paths = [
    (
        'Full, {clf}',
        find_newest_data_path(f'tcga_train_response_stratify_alpha_{args.alpha:.2f}'),
    ),
    (
        'Full + Cell Lines, {clf}',
        find_newest_data_path(f'dual_model_train_alpha_{args.alpha:.2f}'),
    ),
    (
        'UPMC Patients, {clf}',
        find_newest_data_path(f'tcga_train_upmc_response_stratify_alpha_{args.alpha:.2f}'),
    ),
    (
        'Naïve Features, {clf}',
        find_newest_data_path(f'tcga_train_basic_response_stratify_alpha_{args.alpha:.2f}'),
    ),
    (
        'UPMC Patients, Naïve Features, {clf}',
        find_newest_data_path(f'tcga_train_basic_upmc_response_stratify_alpha_{args.alpha:.2f}'),
    ),
]

drugs = ['ai_all', 'arimidex']
selected_clfs = ['rf', 'svm']
trimmed_clfs = {selected_clfs[0]}
trimmed_crossval_labels = {cip[0] for cip in crossval_input_paths[:3]}

def load_roc_data_single(path: Path, drug: str) -> RocData:
    with open(path / f'roc_data_{drug}.pickle', 'rb') as f:
        return pickle.load(f)

def load_l1o_data_compute_roc(path: Path, drug: str) -> Dict[str, RocData]:
    with pd.HDFStore(path / f'l1o_preds_{drug}.hdf5', 'r') as store:
        labels = store['labels']
        preds = store['l1o_preds']

    return {
        clf: RocData.calculate(labels, preds.loc[:, clf])
        for clf in preds.columns
    }

def auc(lrd: Tuple[str, RocData]) -> float:
    return lrd[1].auc

for drug in drugs:
    desc = drug_name_to_description(drug, title=True)
    all_roc_data: List[Tuple[str, RocData]] = []
    trimmed_roc_data: List[Tuple[str, RocData]] = []

    for label, path in single_curve_input_paths:
        try:
            roc_data = (label, load_roc_data_single(path, drug))
            all_roc_data.append(roc_data)
            trimmed_roc_data.append(roc_data)
        except FileNotFoundError:
            pass

    for label_template, path in crossval_input_paths:
        try:
            print('Loading data from', path)
            all_clf_roc_data = load_l1o_data_compute_roc(path, drug)
            for clf in selected_clfs:
                label = label_template.format(clf=clf.upper())
                roc_data = (label, all_clf_roc_data[clf])
                all_roc_data.append(roc_data)
                if clf in trimmed_clfs and label_template in trimmed_crossval_labels:
                    trimmed_roc_data.append(roc_data)
        except OSError:
            # Some analysis types don't exist, like Arimidex-only for the dual model
            pass

    all_sorted_roc = sorted(all_roc_data, key=auc, reverse=True)
    trimmed_sorted_roc = sorted(trimmed_roc_data, key=auc, reverse=True)

    with new_plot():
        plt.figure(figsize=CROSSVAL_FIGSIZE)
        for label, rd in all_sorted_roc:
            plt.plot(rd.fpr, rd.tpr, lw=1, label=f'{label} (area = {rd.auc:.{SIGNIFICANT_DIGITS}f})')

        plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

        plt.xlim([-0.01, 1.01])
        plt.ylim([-0.01, 1.01])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.axes().set_aspect('equal', 'datalim')
        plt.legend(loc='lower right')
        plt.title(f'ROC Comparison: {desc}')

        figure_path = output_path / f'roc_comparison_all_{drug}.pdf'
        plt.savefig(figure_path, bbox_inches='tight')

    with new_plot():
        plt.figure(figsize=CROSSVAL_FIGSIZE)
        for label, rd in trimmed_sorted_roc:
            plt.plot(rd.fpr, rd.tpr, lw=1, label=f'{label} (area = {rd.auc:.{SIGNIFICANT_DIGITS}f})')

        plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

        plt.xlim([-0.01, 1.01])
        plt.ylim([-0.01, 1.01])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.axes().set_aspect('equal', 'datalim')
        plt.legend(loc='lower right')
        plt.title(f'ROC Comparison: {desc}')

        figure_path = output_path / f'roc_comparison_trimmed_{drug}.pdf'
        plt.savefig(figure_path, bbox_inches='tight')
