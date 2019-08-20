#!/usr/bin/env python3
from collections import Counter, defaultdict
from itertools import chain
import json
from pathlib import Path
import pickle

from data_path_utils import (
    DATA_PATH,
    create_data_path,
    find_newest_data_path,
)
import numpy as np
import pandas as pd

from propagation import normalize, propagate
from utils import weighted_correlation

data_path = create_data_path('drug_targets')

with Path('~/data/drugs_targets.json').expanduser().open() as f:
    raw_data = json.load(f)

drug_target_data_all = pd.DataFrame(raw_data).T
# Select those with protein targets
drug_target_data = drug_target_data_all.loc[drug_target_data_all.gene_symbols.notnull(), :]

dtd_path = data_path / 'drug_targets.pickle'
print('Saving drug target data matrix to', dtd_path)
drug_target_data.to_pickle(dtd_path)

synonyms = defaultdict(list)
for row_name, synonym_csv in drug_target_data.synonyms.iteritems():
    for synonym in synonym_csv.split(','):
        synonyms[synonym].append(row_name)

synonym_counts = pd.Series({k: len(v) for k, v in synonyms.items()})

with (find_newest_data_path('build_hippie_network') / 'network.pickle').open('rb') as f:
    network = pickle.load(f)
print('Loaded network')

nodes = sorted(network.nodes())
node_set = set(nodes)

w_prime = normalize(network)

all_targets = set(chain.from_iterable(drug_target_data.gene_symbols))

both = node_set & all_targets

print('Nodes in network:', len(node_set))
print('Genes targeted by at least one drug:', len(all_targets))
print('Overlap between network and targets:', len(both))

print('Genes targeted by drugs but missing from network:')
missing = all_targets - node_set
for gene in missing:
    print(gene)

def get_prop_vec(genes):
    s = pd.Series(0.0, index=nodes)
    genes_in_network = list(set(genes) & node_set)
    s.loc[genes_in_network] = 1
    return s

proximity_mat = pd.DataFrame(0.0, index=drug_target_data.index, columns=nodes)

for row, genes in drug_target_data.gene_symbols.iteritems():
    print('Running network smoothing for drug', row)
    vec = get_prop_vec(genes)
    prop_vec = propagate(w_prime, vec, verbose=False)
    proximity_mat.loc[row] = prop_vec

pm_path = data_path / 'proximity_matrix.pickle'
print('Saving proximity matrix to', pm_path)
proximity_mat.to_pickle(pm_path)

mutations = pd.read_pickle(DATA_PATH / 'mutations.pickle')

prop_muts = pd.read_pickle(DATA_PATH / 'alpha_0.80/mutations_propagated.pickle')
prop_diffexpr = pd.read_pickle(DATA_PATH / 'alpha_0.80/diffexpr_propagated.pickle')

corr_scores_mut = pd.DataFrame(0.0, index=prop_muts.index, columns=drug_target_data.index)
corr_scores_expr = pd.DataFrame(0.0, index=prop_muts.index, columns=drug_target_data.index)

print('Assigning correlation scores')


for drug_id in drug_target_data.index:
    d = proximity_mat.loc[drug_id, :]
    di = d / d.sum()
    for sample in prop_muts.index:
        m = prop_muts.loc[sample, :]
        if np.allclose(m.sum(), 0):
            continue
        w = np.vstack([m / m.sum(), di]).min(axis=0)
        if np.allclose(w, 0):
            print('No overlap in drug and mutation proximity for', sample, 'and', drug_id)
            continue
        c = weighted_correlation(d, m, w)
        corr_scores_mut.loc[sample, drug_id] = c
    for sample in prop_diffexpr.index:
        e = prop_diffexpr.loc[sample, :]
        if np.allclose(e.sum(), 0):
            continue
        w = np.vstack([e / e.sum(), di]).min(axis=0)
        if np.allclose(w, 0):
            print('No overlap in drug and diffexpr proximity for', sample, 'and', drug_id)
            continue
        c = weighted_correlation(d, e, w)
        corr_scores_expr.loc[sample, drug_id] = c

csm_path = data_path / 'corr_scores_mut.pickle'
print('Saving mutation correlation scores to', csm_path)
corr_scores_mut.to_pickle(csm_path)

cse_path = data_path / 'corr_scores_expr.pickle'
print('Saving diffexpr correlation scores to', cse_path)
corr_scores_expr.to_pickle(cse_path)

with_corr_scores = (corr_scores_mut != 0).any(axis=1)
samples_with_corr_scores = prop_muts.index[with_corr_scores]

max_corr_drug_by_sample = corr_scores_mut.loc[samples_with_corr_scores, :].idxmax(axis=1)

drug_names = drug_target_data.preferredCompoundName[max_corr_drug_by_sample]
drug_counts = Counter(drug_names)
