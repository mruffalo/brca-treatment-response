#!/usr/bin/env python3
from argparse import ArgumentParser
import pickle

from data_path_utils import (
    create_data_path,
    create_output_path,
    find_newest_data_path,
)
import numpy as np
import pandas as pd

from constants import drugs, targets
from propagation import normalize, propagate
from utils import (
    DEFAULT_ALPHA,
    consolidate_mut_expr_scores,
    element_wise_min,
    weighted_correlation,
)

selected_cancer = 'brca'

p = ArgumentParser()
p.add_argument('--alpha', type=float, default=DEFAULT_ALPHA)

if __name__ == '__main__':
    args = p.parse_args()
else:
    args = p.parse_args([])

label = f'treatment_features_alpha_{args.alpha:.2f}'
data_path = create_data_path(label)
output_path = create_output_path(label)

network_path = find_newest_data_path('build_hippie_network')
with (network_path / 'network.pickle').open('rb') as f:
    network = pickle.load(f)
nodes = sorted(network.nodes())
node_set = set(nodes)

w_prime = normalize(network)

def get_prop_vec(name, genes):
    s = pd.Series(0.0, index=nodes)
    gene_set = set(genes)
    genes_in_network = gene_set & node_set
    genes_not_in_network = gene_set - node_set
    print(
        'Drug {}: {} genes in network, {} not'.format(
            name,
            len(genes_in_network),
            len(genes_not_in_network),
        )
    )
    s.loc[list(genes_in_network)] = 1
    return s

proximity_mat = pd.DataFrame(0.0, index=drugs, columns=nodes)

for name, genes in targets.items():
    print('Running network smoothing for drug', name)
    vec = get_prop_vec(name, genes)
    prop_vec = propagate(w_prime, vec, verbose=False, alpha=args.alpha)
    proximity_mat.loc[name, :] = prop_vec

pm_path = data_path / '{}_drug_target_proximity.pickle'.format(selected_cancer)
print('Saving proximity matrix to', pm_path)
proximity_mat.to_pickle(pm_path)

prop_data_path = find_newest_data_path(f'propagate_mutations_alpha_{args.alpha:.2f}')
with pd.HDFStore(prop_data_path / 'data_propagated.hdf5', 'r') as store:
    prop_muts = store['muts']
    prop_diffexpr = store['diffexpr']

full_scores_mut = {}
full_scores_diffexpr = {}

corr_scores_mut = pd.DataFrame(0.0, index=prop_muts.index, columns=drugs)
corr_scores_diffexpr = pd.DataFrame(0.0, index=prop_diffexpr.index, columns=drugs)

mean_scores_mut = pd.DataFrame(0.0, index=prop_muts.index, columns=drugs)
mean_scores_diffexpr = pd.DataFrame(0.0, index=prop_diffexpr.index, columns=drugs)

std_scores_mut = pd.DataFrame(0.0, index=prop_muts.index, columns=drugs)
std_scores_diffexpr = pd.DataFrame(0.0, index=prop_diffexpr.index, columns=drugs)

for drug_id in drugs:
    print('Assigning feature scores for', drug_id)
    full_scores_mut[drug_id] = pd.DataFrame(
        0.0,
        index=prop_muts.index,
        columns=prop_muts.columns,
    )
    full_scores_diffexpr[drug_id] = pd.DataFrame(
        0.0,
        index=prop_diffexpr.index,
        columns=prop_diffexpr.columns,
    )

    d = proximity_mat.loc[drug_id, :]
    di = d / d.sum()
    for sample in prop_muts.index:
        m = prop_muts.loc[sample, :]
        mi = m / m.sum()
        if np.allclose(m.sum(), 0):
            continue
        w = element_wise_min(mi, di)
        full_scores_mut[drug_id].loc[sample, :] = w
        mean_scores_mut.loc[sample, drug_id] = w.mean()
        std_scores_mut.loc[sample, drug_id] = w.std()
        if np.allclose(w, 0):
            print('No overlap in drug and mutation proximity for', sample, 'and', drug_id)
            continue
        c = weighted_correlation(d, m, w)
        corr_scores_mut.loc[sample, drug_id] = c
    for sample in prop_diffexpr.index:
        e = prop_diffexpr.loc[sample, :]
        ei = e / e.sum()
        if np.allclose(e.sum(), 0):
            continue
        w = element_wise_min(ei, di)
        full_scores_diffexpr[drug_id].loc[sample, :] = w
        mean_scores_diffexpr.loc[sample, drug_id] = w.mean()
        std_scores_diffexpr.loc[sample, drug_id] = w.std()
        if np.allclose(w, 0):
            print('No overlap in drug and diffexpr proximity for', sample, 'and', drug_id)
            continue
        c = weighted_correlation(d, e, w)
        corr_scores_diffexpr.loc[sample, drug_id] = c

full_mut_score_path = data_path / 'full_scores_mut.hdf5'
print('Saving full mut scores to', full_mut_score_path)
with pd.HDFStore(str(full_mut_score_path)) as store:
    for drug, scores in full_scores_mut.items():
        store[drug] = scores

full_diffexpr_score_path = data_path / 'full_scores_diffexpr.hdf5'
print('Saving full diffexpr scores to', full_diffexpr_score_path)
with pd.HDFStore(str(full_diffexpr_score_path)) as store:
    for drug, scores in full_scores_diffexpr.items():
        store[drug] = scores

min_corrs_full = consolidate_mut_expr_scores(corr_scores_mut, corr_scores_diffexpr)
min_means_full = consolidate_mut_expr_scores(mean_scores_mut, mean_scores_diffexpr)
min_stds_full = consolidate_mut_expr_scores(std_scores_mut, std_scores_diffexpr)

min_score_path = data_path / 'min_scores_full.hdf5'
print('Saving consolidated scores to', min_score_path)
with pd.HDFStore(str(min_score_path)) as store:
    store['corr'] = min_corrs_full
    store['mean'] = min_means_full
    store['std'] = min_stds_full
