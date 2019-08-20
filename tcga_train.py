#!/usr/bin/env python3
from argparse import ArgumentParser
from functools import partial
import os
import pickle

from data_path_utils import (
    create_data_output_paths,
    find_newest_data_path,
)
import matplotlib
if '__file__' in globals() or 'SSH_CONNECTION' in os.environ:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    roc_curve,
)
from sklearn.model_selection import LeaveOneOut, StratifiedKFold

from utils import (
    ClassifierData,
    CROSSVAL_FIGSIZE,
    DEFAULT_ALPHA,
    SIGNIFICANT_DIGITS,
    PrData,
    RocData,
    drug_name_to_description,
    new_plot,
    plot_cdf,
    plot_crossval_pr,
    plot_crossval_roc,
    plot_rf_feature_importance,
    save_coefs,
    sorted_intersection,
)

p = ArgumentParser()
p.add_argument('--alpha', type=float, default=DEFAULT_ALPHA)

if '__file__' in globals():
    args = p.parse_args()
else:
    args = p.parse_args([])

output_label = f'tcga_train_response_stratify_alpha_{args.alpha:.2f}'
data_path, output_path = create_data_output_paths(output_label)

clf_factories = {
    'lr': LogisticRegression,
    'lr_lasso': partial(LogisticRegression, penalty='l1', solver='liblinear', C=10),
    'rf': partial(RandomForestClassifier, n_estimators=100),
    'svm': partial(svm.SVC, probability=True),
}

selected_clfs = {'rf', 'svm'}

fold_count = 5

def plot_univariate_feature_roc_auc(
        drug_name: str,
        drug_data_matrix: pd.DataFrame,
        labels: pd.Series,
        filename_extra: str='',
        top_bottom_count: int=20,
):
    aucs = pd.Series(0.0, index=drug_data_matrix.columns)
    for col in drug_data_matrix:
        fpr, tpr, thresholds = roc_curve(labels, drug_data_matrix.loc[:, col])
        roc_auc = auc(fpr, tpr)
        aucs.loc[col] = roc_auc

    aucs_sorted = aucs.sort_values()
    sel = pd.Series(False, index=aucs_sorted.index)
    sel.iloc[:top_bottom_count] = True
    sel.iloc[-top_bottom_count:] = True

    aucs_selected = aucs_sorted.loc[sel]
    col = ['C1'] * top_bottom_count + ['C0'] * top_bottom_count

    with new_plot():
        fig, ax = plt.subplots(figsize=(7, 9))
        ax.set_xlim([0, 1])
        aucs_selected.plot.barh(ax=ax, color=col)
        ax.axvline(0.5, linestyle='--', alpha=0.25, color='black')

        desc = drug_name_to_description(drug_name)
        ax.set_title(f'Univariate feature ROC AUC: {desc}')

        if filename_extra:
            filename_extra = f'_{filename_extra}'
        filename = f'{drug_name}_univariate_roc_auc{filename_extra}.pdf'
        figure_path = output_path / filename
        print('Saving univariate ROC AUC to', figure_path)
        fig.savefig(figure_path, bbox_inches='tight')

selected_drugs = ['arimidex', 'ai_all']

feature_label_path = find_newest_data_path(f'compute_drug_features_labels_alpha_{args.alpha:.2f}')

for drug in selected_drugs:
    skf = StratifiedKFold(n_splits=fold_count, shuffle=True)

    clfs = {clf_desc: [] for clf_desc in clf_factories}
    trn_roc_data = {clf_desc: [] for clf_desc in clf_factories}
    trn_pr_data = {clf_desc: [] for clf_desc in clf_factories}
    roc_data = {clf_desc: [] for clf_desc in clf_factories}
    pr_data = {clf_desc: [] for clf_desc in clf_factories}

    drug_feature_matrix = pd.read_pickle(feature_label_path / f'feature_matrix_{drug}.pickle')
    switched_from_drug_or_died = pd.read_pickle(feature_label_path / f'labels_{drug}.pickle')

    print('Shape of feature matrix:', drug_feature_matrix.shape)

    plot_univariate_feature_roc_auc(drug, drug_feature_matrix, switched_from_drug_or_died)

    folds = list(skf.split(drug_feature_matrix, switched_from_drug_or_died))

    train_test_split_path = data_path / f'train_test_split_{drug}.pickle'
    print('Saving cross-validation train/test split to', train_test_split_path)
    with open(train_test_split_path, 'wb') as f:
        pickle.dump(folds, f)

    for i, (trn_index, val_index) in enumerate(folds):
        print(f'Performing cross-validation fold {i} for {drug}')
        clf_data = ClassifierData(
            drug_feature_matrix.iloc[trn_index, :].fillna(0),
            switched_from_drug_or_died.iloc[trn_index],
            drug_feature_matrix.iloc[val_index, :].fillna(0),
            switched_from_drug_or_died.iloc[val_index],
        )

        for clf_desc, clf_factory in clf_factories.items():
            clf_obj = clf_factory()
            clfs[clf_desc].append(clf_obj)

            clf_obj.fit(clf_data.trn_matrix, clf_data.trn_labels)

            coef_path = data_path / f'fold_{i}_{drug}_{clf_desc}_coefs.csv'
            save_coefs(clf_obj, coef_path, clf_data.trn_matrix.columns)

            rf_feature_path = output_path / f'{drug}_rf_features_fold_{i}.pdf'
            plot_rf_feature_importance(clf_obj, rf_feature_path, clf_data.trn_matrix.columns)

            trn_preds = clf_obj.predict_proba(clf_data.trn_matrix)
            trn_preds_series = pd.Series(
                trn_preds[:, 1],
                index=clf_data.trn_matrix.index,
                name=f'switch_{clf_desc}_pred_{drug}',
            )

            trn_rd = RocData.calculate(clf_data.trn_labels, trn_preds_series)
            trn_rd.save(data_path / f'trn_roc_data_{drug}_{clf_desc}_fold_{i}.pickle')
            trn_roc_data[clf_desc].append(trn_rd)
            trn_pr_data[clf_desc].append(PrData.calculate(clf_data.trn_labels, trn_preds_series))

            preds = clf_obj.predict_proba(clf_data.val_matrix)
            preds_series = pd.Series(
                preds[:, 1],
                index=clf_data.val_matrix.index,
                name=f'switch_{clf_desc}_pred_{drug}',
            )

            rd = RocData.calculate(clf_data.val_labels, preds_series)
            rd.save(data_path / f'roc_data_{drug}_{clf_desc}_fold_{i}.pickle')
            roc_data[clf_desc].append(rd)
            pr_data[clf_desc].append(PrData.calculate(clf_data.val_labels, preds_series))

    desc = drug_name_to_description(drug, title=True)

    for clf_desc, rd_list in trn_roc_data.items():
        plot_crossval_roc(
            rd_list,
            f'{desc}: {clf_desc.upper()} Training ROC',
            output_path / f'trn_{drug}_{clf_desc}_crossval_roc.pdf',
        )

    for clf_desc, pr_list in trn_pr_data.items():
        plot_crossval_pr(
            pr_list,
            f'{desc}: {clf_desc.upper()} Training P-R',
            output_path / f'trn_{drug}_{clf_desc}_crossval_pr.pdf',
        )

    for clf_desc, rd_list in roc_data.items():
        plot_crossval_roc(
            rd_list,
            f'{desc}: {clf_desc.upper()} Prediction ROC',
            output_path / f'{drug}_{clf_desc}_crossval_roc.pdf',
        )

    for clf_desc, pr_list in pr_data.items():
        plot_crossval_pr(
            pr_list,
            f'{desc}: {clf_desc.upper()} Prediction Prediction Precision-Recall',
            output_path / f'{drug}_{clf_desc}_crossval_pr.pdf',
        )

    # Leave-one-out
    l1o_preds = pd.DataFrame(0.0, index=switched_from_drug_or_died.index, columns=sorted(clf_factories))
    l1o = LeaveOneOut()

    l10_clfs = {clf_desc: [] for clf_desc in clf_factories}
    l10_trn_roc_data = {clf_desc: [] for clf_desc in clf_factories}
    l10_trn_pr_data = {clf_desc: [] for clf_desc in clf_factories}
    l10_roc_data = {clf_desc: [] for clf_desc in clf_factories}
    l10_pr_data = {clf_desc: [] for clf_desc in clf_factories}

    for i, (trn_index, val_index) in enumerate(l1o.split(drug_feature_matrix)):
        print(f'Performing L1O split {i} for {drug}')
        clf_data = ClassifierData(
            drug_feature_matrix.iloc[trn_index, :].fillna(0),
            switched_from_drug_or_died.iloc[trn_index],
            drug_feature_matrix.iloc[val_index, :].fillna(0),
            switched_from_drug_or_died.iloc[val_index],
        )

        for clf_desc, clf_factory in clf_factories.items():
            clf_obj = clf_factory()
            l10_clfs[clf_desc].append(clf_obj)

            clf_obj.fit(clf_data.trn_matrix, clf_data.trn_labels)

            # coef_path = data_path / f'fold_{i}_{drug}_{clf_desc}_coefs.csv'
            # save_coefs(clf_obj, coef_path, clf_data.trn_matrix.columns)

            # rf_feature_path = output_path / f'{drug}_rf_features.pdf'
            # plot_rf_feature_importance(clf_obj, rf_feature_path, clf_data.trn_matrix.columns)

            trn_preds = clf_obj.predict_proba(clf_data.trn_matrix)
            trn_preds_series = pd.Series(
                trn_preds[:, 1],
                index=clf_data.trn_matrix.index,
                name=f'switch_{clf_desc}_pred_{drug}',
            )

            trn_rd = RocData.calculate(clf_data.trn_labels, trn_preds_series)
            # trn_rd.save(data_path / f'trn_roc_data_{drug}_{clf_desc}_fold_{i}.pickle')
            l10_trn_roc_data[clf_desc].append(trn_rd)
            l10_trn_pr_data[clf_desc].append(PrData.calculate(clf_data.trn_labels, trn_preds_series))

            preds = clf_obj.predict_proba(clf_data.val_matrix)
            preds_series = pd.Series(
                preds[:, 1],
                index=clf_data.val_matrix.index,
                name=f'switch_{clf_desc}_pred_{drug}',
            )

            # one sample
            l1o_preds.loc[preds_series.index[0], clf_desc] = preds_series.iloc[0]

    # TODO improve this
    label_matrix = pd.DataFrame({col: switched_from_drug_or_died for col in sorted(clf_factories)})

    sq_error = (l1o_preds - label_matrix) ** 2

    with new_plot():
        fig, ax = plt.subplots()
        sq_error.plot.hist(bins=32, alpha=0.75, ax=ax)

        figure_path = output_path / f'{drug}_l1o_sq_error_hist.pdf'
        print('Saving L1O sq. error histogram to', figure_path)
        fig.savefig(figure_path, bbox_inches='tight')

    with new_plot():
        fig, ax = plt.subplots()

        for col in sq_error.columns:
            plot_cdf(sq_error.loc[:, col], ax=ax)

        ax.legend()

        figure_path = output_path / f'{drug}_l1o_sq_error_cdf.pdf'
        print('Saving L1O sq. error CDF to', figure_path)
        fig.savefig(figure_path, bbox_inches='tight')

    l1o_rds = [
        (clf_desc, RocData.calculate(switched_from_drug_or_died, l1o_preds.loc[:, clf_desc]))
        for clf_desc in sorted(clf_factories)
    ]

    with new_plot():
        plt.figure(figsize=CROSSVAL_FIGSIZE)

        for clf, rd in l1o_rds:
            label = f'{clf.upper()} L1O overall ROC (area = {rd.auc:.{SIGNIFICANT_DIGITS}f})'
            plt.plot(rd.fpr, rd.tpr, lw=1, label=label)

        plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

        plt.xlim([-0.01, 1.01])
        plt.ylim([-0.01, 1.01])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.axes().set_aspect('equal', 'datalim')
        plt.legend(loc='lower right')
        title = f'{desc}: Leave-One-Out Cross-Validation ROC'
        plt.title(title)

        figure_path = output_path / f'{drug}_l1o_crossval_roc.pdf'
        plt.savefig(figure_path, bbox_inches='tight')

    with new_plot():
        plt.figure(figsize=CROSSVAL_FIGSIZE)

        for clf, rd in l1o_rds:
            if clf in selected_clfs:
                label = f'{clf.upper()} L1O overall ROC (area = {rd.auc:.{SIGNIFICANT_DIGITS}f})'
                plt.plot(rd.fpr, rd.tpr, lw=1, label=label)

        plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

        plt.xlim([-0.01, 1.01])
        plt.ylim([-0.01, 1.01])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.axes().set_aspect('equal', 'datalim')
        plt.legend(loc='lower right')
        title = f'{desc}: Leave-One-Out Cross-Validation ROC'
        plt.title(title)

        figure_path = output_path / f'{drug}_l1o_crossval_trimmed_roc.pdf'
        plt.savefig(figure_path, bbox_inches='tight')

    l1o_pred_path = data_path / f'l1o_preds_{drug}.hdf5'
    print('Saving L1O preds to', l1o_pred_path)
    with pd.HDFStore(l1o_pred_path) as store:
        store['l1o_preds'] = l1o_preds
        store['labels'] = switched_from_drug_or_died
        store['sq_error'] = sq_error
