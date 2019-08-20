from contextlib import contextmanager
from itertools import cycle, islice, zip_longest
from math import factorial
from os import environ
from pathlib import Path
import pickle
import re
from typing import Dict, Hashable, Iterable, List, Optional, Sequence, Tuple, TypeVar

from data_path_utils import append_to_filename
import matplotlib
if 'SSH_CONNECTION' in environ:
    matplotlib.use('Agg')
from matplotlib.colors import Colormap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interp
import scipy.sparse
from sklearn.metrics import (
    auc,
    precision_recall_curve,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler

from convert_labels import convert_label

DEFAULT_ALPHA = 0.8

CROSSVAL_FIGSIZE = (8, 8)

SIGNIFICANT_DIGITS = 2

SLURM_SUBMIT_COMMAND_TEMPLATE = [
    'sbatch',
    '{script_path}',
]

T = TypeVar('T')

def grouper(iterable, n, fillvalue=None):
    """
    Collect data into fixed-length chunks or blocks
    """
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)

def roundrobin(*iterables):
    "roundrobin('ABC', 'D', 'EF') --> A D E B F C"
    # Recipe credited to George Sakkis
    pending = len(iterables)
    nexts = cycle(iter(it).__next__ for it in iterables)
    while pending:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            pending -= 1
            nexts = cycle(islice(nexts, pending))

@contextmanager
def new_plot():
    plt.clf()
    try:
        yield
    finally:
        plt.clf()
        plt.close()

def sorted_set_op(items: Iterable[T], func) -> List[T]:
    sets = [set(item) for item in items]
    data = func(*sets)
    return sorted(data)

def sorted_intersection(*items: T) -> List[T]:
    return sorted_set_op(items, set.intersection)

def sorted_union(*items: T) -> List[T]:
    return sorted_set_op(items, set.union)

def strip_prefix(string: str, prefix: str) -> str:
    """
    :param string: String to strip `prefix` from, if present
    :param prefix: Prefix to remove from `string`
    :return:
    """
    if string.startswith(prefix):
        return string[len(prefix):]
    return string

def strip_suffix(string: str, suffix: str) -> str:
    """
    :param string: String to strip `suffix` from, if present
    :param suffix: Suffix to remove from `string`
    :return:
    """
    if string.endswith(suffix):
        return string[:-len(suffix)]
    return string

def first(iterable: Iterable[T]) -> T:
    return next(iter(iterable))

def weighted_correlation(x, y, w):
    cov_mat = np.cov(x, y, aweights=w)
    cov = cov_mat[0, 1]
    corr = cov / np.sqrt(cov_mat[0, 0] * cov_mat[1, 1])
    return corr

def element_wise_attr_func(*vecs, attr_name: str):
    s = np.vstack(vecs)
    return getattr(s, attr_name)(axis=0)

def element_wise_min(*vecs):
    """
    :param vecs: sequence of vectors (probably np.arrays), of same dimension
    :return: A new vector, composed of the minimum of all values in each position
    """
    return element_wise_attr_func(*vecs, attr_name='min')

def element_wise_max(*vecs):
    """
    :param vecs: sequence of vectors (probably np.arrays), of same dimension
    :return: A new vector, composed of the maximum of all values in each position
    """
    return element_wise_attr_func(*vecs, attr_name='max')

def consolidate_data_frames(dfs: Iterable[Tuple[pd.DataFrame, str]]):
    """
    Consolidates DataFrames that have identical indexes, and potentially-identical
    columns. Duplicates each data frame, appends an underscore and the label to the
    name of each column, and then returns a consolidated DataFrame with all columns.
    :param dfs: Iterable of 2-tuples:
      [0] DataFrame
      [1] label
    :return:
    """
    new_dfs = []
    for df_orig, label in dfs:
        df = df_orig.copy()
        if label:
            df.columns = ['{}_{}'.format(c, label) for c in df.columns]
        new_dfs.append(df)
    return pd.concat(new_dfs, axis=1)

def consolidate_mut_expr_scores(mut: pd.DataFrame, expr: pd.DataFrame) -> pd.DataFrame:
    """
    :param mut: Mutation DataFrame
    :param expr: Expression DataFrame
    :return: Concatenated DataFrame with adjusted columns:
      '_mut' is appended to column names from the mutation DF,
      '_expr' is appended to column names from the expression DF.
    """
    return consolidate_data_frames(
        [
            (mut, 'mut'),
            (expr, 'expr'),
        ]
    )

BASE_FPR = np.linspace(0, 1, 101)

class RocData:
    @classmethod
    def calculate(cls, labels, scores):
        fpr, tpr, thresholds = roc_curve(labels, scores)
        tpr_interp = interp(BASE_FPR, fpr, tpr)
        tpr_interp[0] = 0.0
        roc_auc = auc(fpr, tpr)

        self = cls(fpr, tpr, tpr_interp, roc_auc)
        return self

    def __init__(self, fpr, tpr, tpr_interp, roc_auc):
        self.fpr = fpr
        self.tpr = tpr
        self.tpr_interp = tpr_interp
        self.auc = roc_auc

    def save(self, path: Path):
        print('Saving ROC data to', path)
        with open(path, 'wb') as f:
            pickle.dump(self, f)

class PrData:
    @classmethod
    def calculate(cls, labels, scores):
        prec, rec, thresholds = precision_recall_curve(labels, scores)

        self = cls(prec, rec)
        return self

    def __init__(self, prec, rec):
        self.prec = prec
        self.rec = rec

class ClassifierData:
    __slots__ = ['trn_matrix', 'trn_labels', 'val_matrix', 'val_labels']

    def __init__(self, trn_matrix, trn_labels, val_matrix, val_labels):
        assert trn_matrix.shape[0] == trn_labels.shape[0]
        assert val_matrix.shape[0] == val_labels.shape[0]

        self.trn_matrix = trn_matrix
        self.trn_labels = trn_labels
        self.val_matrix = val_matrix
        self.val_labels = val_labels

def scale_continuous_df_cols(df: pd.DataFrame) -> Tuple[StandardScaler, pd.DataFrame]:
    """
    Rescales continuous columns of `df` to mean 0 and standard deviation 1.
    Binary columns (containing only 0 or 1) are returned as-is.
    :param df:
    :return:
    """
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(df.fillna(0)), index=df.index, columns=df.columns)
    zero_one_cols = df.apply(set) <= {0, 1}
    X.loc[:, zero_one_cols] = df.loc[:, zero_one_cols]
    return scaler, X

def choose(n: int, k: int) -> int:
    return int(factorial(n) / (factorial(k) * factorial(n - k)))

def save_coefs(clf_obj, path: Path, all_cols):
    try:
        coefs = clf_obj.coef_
        if not isinstance(coefs, pd.DataFrame):
            coefs = pd.DataFrame(clf_obj.coef_, columns=all_cols)
        print('Saving regression coefficients to', path)
        coefs.T.to_csv(path)
    except (ValueError, AttributeError):
        # Can't just check for the existence of a coef_ attribute; RBF SVMs
        # still have this attribute but even calling hasattr(..., 'coef_')
        # just throws a ValueError
        pass

def plot_rf_feature_importance(clf_obj, path: Path, all_cols):
    if not hasattr(clf_obj, 'feature_importances_'):
        return

    labels = [convert_label(i) for i in all_cols]

    importance = pd.Series(clf_obj.feature_importances_, index=labels)
    importance.sort_values(ascending=False, inplace=True)
    print('Saving random forest feature importance plot to', path)
    with new_plot():
        width = importance.shape[0] // 4
        plt.figure(figsize=(width, 8))
        importance.plot.bar(color='C0')
        plt.title('Random forest feature importance')

        plt.savefig(path, bbox_inches='tight')

    subset_path = append_to_filename(path, '_subset')
    importance_subset = importance.iloc[:32][::-1]
    print('Saving random forest feature importance subset plot to', subset_path)
    with new_plot():
        plt.figure(figsize=(7, 6))
        importance_subset.plot.barh(color='C0')
        plt.title('Random forest feature importance')

        plt.savefig(subset_path, bbox_inches='tight')

Pairs = Sequence[Tuple[Hashable, Hashable]]

def pairs_and_values_to_dataframe(pairs: Pairs, values) -> pd.DataFrame:
    """
    :param pairs: List of 2-tuples, each element of which must be hashable.
     [0] is used for rows, [1] is used for columns
    :param values: Data to use in matrix
    :return: Dense DataFrame built from rows and columns in 'pairs' and data in 'values'
    """
    # TODO: don't duplicate for rows/cols
    rows = sorted(set(pair[0] for pair in pairs))
    cols = sorted(set(pair[1] for pair in pairs))
    row_indexes = {row: i for (i, row) in enumerate(rows)}
    col_indexes = {col: i for (i, col) in enumerate(cols)}
    data_rows = [row_indexes[pair[0]] for pair in pairs]
    data_cols = [col_indexes[pair[1]] for pair in pairs]
    matrix = scipy.sparse.coo_matrix(
        (values, (data_rows, data_cols)),
        shape=(len(rows), len(cols))
    )
    d = pd.DataFrame(matrix.todense(), index=rows, columns=cols)
    return d

def pairs_to_zero_one_dataframe(pairs: Pairs) -> pd.DataFrame:
    """
    :param pairs: List of 2-tuples, each element of which must be hashable.
     [0] is used for rows, [1] is used for columns
    :return: 0/1 DataFrame
    """
    values = np.ones(len(pairs))
    df = pairs_and_values_to_dataframe(pairs, values)
    # When parsing MAF files, might have duplicate mutations; coerce to bool and back to float
    return df.astype(bool).astype(float)

def plot_roc(rd: RocData, title: str, figure_path: Path):
    with new_plot():
        plt.figure(figsize=CROSSVAL_FIGSIZE)

        plt.plot(rd.fpr, rd.tpr, lw=1, label=f'ROC (area = {rd.auc:.{SIGNIFICANT_DIGITS}f})')

        plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

        plt.xlim([-0.01, 1.01])
        plt.ylim([-0.01, 1.01])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.axes().set_aspect('equal', 'datalim')
        plt.legend(loc='lower right')
        plt.title(title)

        plt.savefig(figure_path, bbox_inches='tight')

def plot_pr(pr: PrData, title: str, figure_path: Path):
    with new_plot():
        plt.figure(figsize=CROSSVAL_FIGSIZE)

        plt.plot(pr.rec, pr.prec, lw=1)

        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.axes().set_aspect('equal', 'datalim')
        plt.title(title)

        plt.savefig(figure_path, bbox_inches='tight')

def plot_crossval_roc(rd_list: List[RocData], title: str, figure_path: Path):
    with new_plot():
        plt.figure(figsize=CROSSVAL_FIGSIZE)

        for i, rd in enumerate(rd_list):
            plt.plot(rd.fpr, rd.tpr, lw=1, label=f'ROC fold {i} (area = {rd.auc:.{SIGNIFICANT_DIGITS}f})')

        roc_aucs = np.array([rd.auc for rd in rd_list])

        tp_interp_all = np.array([rd.tpr_interp for rd in rd_list])
        tp_interp_mean = tp_interp_all.mean(axis=0)
        plt.plot(
            BASE_FPR,
            tp_interp_mean,
            lw=2,
            label=f'ROC mean (area = {roc_aucs.mean():.{SIGNIFICANT_DIGITS}f})',
            color='black',
        )

        plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

        plt.xlim([-0.01, 1.01])
        plt.ylim([-0.01, 1.01])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.axes().set_aspect('equal', 'datalim')
        plt.legend(loc='lower right')
        plt.title(title)

        plt.savefig(figure_path, bbox_inches='tight')

def plot_crossval_pr(pr_list: List[PrData], title: str, figure_path: Path):
    with new_plot():
        plt.figure(figsize=CROSSVAL_FIGSIZE)

        for i, pr in enumerate(pr_list):
            plt.plot(pr.rec, pr.prec, lw=1, label=f'PR fold {i}')

        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.axes().set_aspect('equal', 'datalim')
        plt.legend(loc='lower right')
        plt.title(title)

        plt.savefig(figure_path, bbox_inches='tight')

def dataframe_pcolormesh(matrix: pd.DataFrame, cmap: Optional[Colormap]=None, x_label=True, **kwargs):
    """
    Plots a data frame as an image, using the specified color map to transform values.

    It is the user's responsibility to save the resulting plot.
    """
    if x_label:
        x_figsize = matrix.shape[1] * (1 / 4)
        tick_axes = 'both'
    else:
        x_figsize = matrix.shape[1] / 15
        tick_axes = 'x'

    plt.figure(
        figsize=(x_figsize, matrix.shape[0] / 5),
    )

    ax = plt.gca()

    ax.tick_params(axis=tick_axes, direction='out')

    if x_label:
        ax.set_xticks(range(matrix.shape[1]))
        ax.xaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
        ax.xaxis.set_minor_formatter(matplotlib.ticker.FixedFormatter(matrix.columns))
        x_minor_locator = matplotlib.ticker.AutoMinorLocator(n=2)
        x_minor_locator.MAXTICKS = 1000000000
        ax.xaxis.set_minor_locator(x_minor_locator)
        for tick in ax.xaxis.get_minor_ticks():
            tick.tick1line.set_markersize(0)
            tick.tick2line.set_markersize(0)
            tick.label1.set_horizontalalignment('center')
            tick.label1.set_rotation('vertical')

    ax.set_yticks(range(matrix.shape[0]))
    ax.yaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
    ax.yaxis.set_minor_formatter(matplotlib.ticker.FixedFormatter(matrix.index))
    y_minor_locator = matplotlib.ticker.AutoMinorLocator(n=2)
    y_minor_locator.MAXTICKS = 1000000000
    ax.yaxis.set_minor_locator(y_minor_locator)
    for tick in ax.yaxis.get_minor_ticks():
        tick.tick1line.set_markersize(0)
        tick.tick2line.set_markersize(0)
        tick.label1.set_verticalalignment('center')

    if cmap is not None:
        kwargs['cmap'] = cmap
    mesh = ax.pcolormesh(matrix, **kwargs)

    plt.colorbar(mesh, ax=ax)

    ax.invert_yaxis()

def plot_cdf(s: pd.Series, ax=None):
    s_sorted = s.sort_values()
    n = s.shape[0]
    f = np.arange(n, dtype=float) / n
    if ax is None:
        ax = plt.gca()
    ax.plot(s_sorted, f, label=s.name)

def to_matplotlib_sci_notation(value: float, digits=4) -> str:
    initial_str = f'{value:.{digits}e}'
    pieces = initial_str.split('e')
    return f'{pieces[0]} \\times 10^{{{pieces[1]}}}'

def drug_name_to_description(drug_name: str, title=False) -> str:
    """
    Standard way to convert drug names to "switch from {drug}", or
    "all aromatase inhibitor response"
    """
    if drug_name == 'ai_all':
        s = 'Response to All Aromatase Inhibitors'
    else:
        s = f'Switch from {drug_name.title()}'

    if title:
        return s
    else:
        return s.lower()

del T
