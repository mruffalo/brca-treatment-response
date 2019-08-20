#!/usr/bin/env python3
from pathlib import Path

from data_path_utils import create_data_path
import pandas as pd

cell_line_expr_path = Path('~/data/brca-cell-lines/pmid26771497/breast_rnaseq_qn.txt').expanduser()

expr_data_raw = pd.read_table(cell_line_expr_path, index_col='gene_id')
# Index is Entrez ID, first column is HUGO symbol, second is ensembl ID
expr_data = expr_data_raw.iloc[:, 2:]
expr_data.columns = [col.upper() for col in expr_data.columns]
expr_data = expr_data.T

data_path = create_data_path('parse_pmid26771497_expr')
expr_path = data_path / 'expr.pickle'
print('Saving expression data to', expr_path)
expr_data.to_pickle(expr_path)
