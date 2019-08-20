#!/usr/bin/env python3
from argparse import ArgumentParser
from multiprocessing import Pool
import pickle

from data_path_utils import (
    create_data_path,
    find_newest_data_path,
)
import numpy as np
import pandas as pd

from gene_mappings import read_hugo_entrez_mapping
from utils import DEFAULT_ALPHA, sorted_intersection
from propagation import propagate, normalize

DEFAULT_SUBPROCESSES = 2

p = ArgumentParser()
p.add_argument('-s', '--subprocesses', type=int, default=DEFAULT_SUBPROCESSES)
p.add_argument('--alpha', type=float, default=DEFAULT_ALPHA)

if __name__ == '__main__':
    args = p.parse_args()
else:
    args = p.parse_args([])

data_path = create_data_path(f'propagate_mutations_alpha_{args.alpha:.2f}')

with (find_newest_data_path('build_hippie_network') / 'network.pickle').open('rb') as f:
    network = pickle.load(f)
print('Loaded network')

w_prime = normalize(network)
node_set = set(network.nodes())
nodes = sorted(node_set)
node_count = len(nodes)

with pd.HDFStore(find_newest_data_path('parse_tcga_mutations') / 'mutations.hdf5', 'r') as store:
    mutations = store['muts']
print('Read mutations')

expr = pd.read_pickle(find_newest_data_path('parse_cosmic_diffexpr') / 'brca_expr.pickle')
print('Read log-fold expression with Hugo symbols')
cutoff = 2
print('Binarizing log-fold expression with cutoff {}'.format(cutoff))
diffexpr_hugo = (expr.abs() > cutoff).astype(float)

hugo_entrez_mapping = read_hugo_entrez_mapping()

diffexpr_hugo_in_mapping = sorted_intersection(diffexpr_hugo.columns, hugo_entrez_mapping)
print(f'{len(diffexpr_hugo_in_mapping)} of {diffexpr_hugo.shape[1]} gene IDs in expr data are in mapping')
diffexpr_overlap = diffexpr_hugo.loc[:, diffexpr_hugo_in_mapping]
new_diffexpr_cols = [hugo_entrez_mapping[col] for col in diffexpr_overlap.columns]
duplicate_col_count = len(new_diffexpr_cols) - len(set(new_diffexpr_cols))
print('Duplicate columns:', duplicate_col_count)
used_entrez_ids = set()
non_dup_col_indices = []
for i, entrez_id in enumerate(new_diffexpr_cols):
    if entrez_id not in used_entrez_ids:
        non_dup_col_indices.append(i)
    used_entrez_ids.add(entrez_id)

diffexpr_overlap.columns = new_diffexpr_cols
diffexpr_overlap_non_dups = diffexpr_overlap.iloc[:, non_dup_col_indices]
diffexpr = diffexpr_overlap_non_dups.loc[:, sorted(diffexpr_overlap_non_dups.columns)]

def propagate_mutations(param_tuple):
    i, sample, label, sample_count, vec = param_tuple
    if not i % 100:
        print(f'{label}: done with {i} samples ({(i * 100) / sample_count:.2f}%)')
    vector = np.matrix(vec).reshape((node_count, 1))
    propagated = propagate(w_prime, vector, alpha=args.alpha, verbose=False)
    return sample, propagated

def propagate_data(data: pd.DataFrame, label: str):
    """
    :param data: Matrix of (samples, genes); network propagation will be run on each row
    :param label: mutations or diffexpr or something
    :return:
    """
    sample_count = len(data.index)
    data_gene_set = set(data.columns)

    common_genes = sorted_intersection(data.columns, node_set)
    common_genes_path = data_path / f'{label}_common_genes.txt'
    print(f'{label}: saving {len(common_genes)} common genes to {common_genes_path}')
    with common_genes_path.open('w') as f:
        for gene in common_genes:
            print(gene, file=f)

    only_mut_genes = sorted(data_gene_set - node_set)
    only_mut_genes_path = data_path / f'{label}_only_mut_genes.txt'
    print(f'{label}: saving {len(only_mut_genes)} data-only genes to {only_mut_genes_path}')
    with only_mut_genes_path.open('w') as f:
        for gene in only_mut_genes:
            print(gene, file=f)

    only_network_genes = sorted(node_set - data_gene_set)
    only_network_genes_path = data_path / '{}_only_network_genes.txt'.format(label)
    print(f'{label}: saving {len(only_network_genes)} network-only genes to {only_network_genes_path}')
    with only_network_genes_path.open('w') as f:
        for gene in only_network_genes:
            print(gene, file=f)

    data_network = pd.DataFrame(0.0, columns=nodes, index=data.index)
    data_propagated = pd.DataFrame(0.0, columns=nodes, index=data.index)
    data_network.loc[:, common_genes] = data.loc[:, common_genes]

    param_generator = (
        (i, sample, label, sample_count, data_network.loc[sample, :])
        for i, sample in enumerate(data_network.index)
    )

    with Pool(args.subprocesses) as pool:
        for sample, propagated in pool.imap_unordered(
                propagate_mutations,
                param_generator,
        ):
            data_propagated.loc[sample, :] = np.array(propagated).reshape((node_count,))

    return data_propagated

prop_muts = propagate_data(mutations, 'mutations')
prop_diffexpr = propagate_data(diffexpr, 'diffexpr')

hdf5_path = data_path / 'data_propagated.hdf5'
print('Saving propagated data to', hdf5_path)
with pd.HDFStore(hdf5_path) as store:
    store['muts'] = prop_muts
    store['diffexpr'] = prop_diffexpr
