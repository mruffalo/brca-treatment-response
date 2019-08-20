#!/usr/bin/env python3
from argparse import ArgumentParser
from math import nan
from pathlib import Path
import pickle
import re

from data_path_utils import (
    DATA_PATH,
    append_to_filename,
    create_data_path,
    find_newest_data_path,
    replace_extension,
)
import pandas as pd
from sklearn.decomposition import PCA

from gene_mappings import read_entrez_hugo_mapping
from pca_component_plot import (
    plot_pca_component_clustering,
    plot_pca_component_subnetwork_cc,
    plot_pca_component_values,
)
from utils import (
    DEFAULT_ALPHA,
    consolidate_data_frames,
    scale_continuous_df_cols,
    sorted_intersection,
)

p = ArgumentParser()
p.add_argument('--alpha', type=float, default=DEFAULT_ALPHA)
p.add_argument('--plot-pca-components', action='store_true')
if '__file__' in globals():
    args = p.parse_args()
else:
    args = p.parse_args([])

entrez_hugo_mapping = read_entrez_hugo_mapping()

output_label = f'compute_drug_features_labels_alpha_{args.alpha:.2f}'
data_path = create_data_path(output_label)

drug_response_dir = find_newest_data_path('drug_response_labels')
tx_info_raw = pd.read_pickle(drug_response_dir / 'tx_info.pickle')

network_path = find_newest_data_path('build_hippie_network') / 'network.pickle'
print('Loading network from', network_path)
with network_path.open('rb') as f:
    network = pickle.load(f)

self_edge_count = 0
# HACK: remove self edges
for node in network.nodes:
    if network.has_edge(node, node):
        network.remove_edge(node, node)
        self_edge_count += 1
print(f'Removed {self_edge_count} self-edges from network')

tcga_clinical_data = pd.read_pickle(drug_response_dir / 'clinical_data_brca.pickle')
tcga_clinical_data.index = tcga_clinical_data.patient_id

upmc_clinical_data = pd.read_csv(DATA_PATH / 'upmc_clinical_data_brca.csv', index_col='SUBJ_ID')
upmc_tx_info = pd.read_csv(DATA_PATH / 'upmc_treatment_info_brca.csv', index_col=0)

upmc_samples = set(upmc_clinical_data.index)
tcga_samples = set(tcga_clinical_data.index) - upmc_samples

all_samples = sorted(tcga_clinical_data.index)

# Used later, to select subsets of patients for cross-validation.
# List of TCGA sample IDs (e.g. TCGA-XX-YYYY)
patient_selection = all_samples

no_value_pattern = re.compile(r'\[Not.*\]')

prop_data_path = find_newest_data_path(f'propagate_mutations_alpha_{args.alpha:.2f}')
with pd.HDFStore(prop_data_path / 'data_propagated.hdf5', 'r') as store:
    prop_muts = store['muts']
    prop_diffexpr = store['diffexpr']

# How many
PCA_COMPONENTS = 10
PCA_COMPONENT_PLOT_COUNT = 50

# !!! TODO consolidate these

pca_muts = PCA(n_components=PCA_COMPONENTS)
prop_muts_pca = pd.DataFrame(
    pca_muts.fit_transform(prop_muts),
    index=prop_muts.index,
    columns=[f'prop_mut_pca_{i}' for i in range(PCA_COMPONENTS)],
)

pca_mut_components = pd.DataFrame(
    pca_muts.components_,
    index=prop_muts_pca.columns,
    columns=prop_muts.columns,
)

pca_muts_path = data_path / 'muts_pca_obj.pickle'
print('Saving muts PCA object to', pca_muts_path)
with open(pca_muts_path, 'wb') as f:
    pickle.dump(pca_muts, f)

if args.plot_pca_components:
    plot_pca_component_values(pca_mut_components, 'mut')
    plot_pca_component_clustering(pca_mut_components, 'mut')
    plot_pca_component_subnetwork_cc(pca_mut_components, 'mut')

pca_mut_components_csv_path = data_path / 'pca_mut_components.csv'
print('Saving PCA mut components to', pca_mut_components_csv_path)
pca_mut_components.T.to_csv(pca_mut_components_csv_path)

pca_mut_components_pickle_path = replace_extension(pca_mut_components_csv_path, 'pickle')
print('Saving PCA mut components to', pca_mut_components_pickle_path)
pca_mut_components.T.to_csv(pca_mut_components_pickle_path)

pca_expr = PCA(n_components=PCA_COMPONENTS)
prop_expr_pca = pd.DataFrame(
    pca_expr.fit_transform(prop_expr),
    index=prop_expr.index,
    columns=[f'prop_expr_pca_{i}' for i in range(PCA_COMPONENTS)],
)

pca_expr_components = pd.DataFrame(
    pca_expr.components_,
    index=prop_expr_pca.columns,
    columns=prop_expr.columns,
)

if args.plot_pca_components:
    plot_pca_component_values(pca_expr_components, 'expr')
    plot_pca_component_clustering(pca_expr_components, 'expr')
    plot_pca_component_subnetwork_cc(pca_expr_components, 'expr')

pca_expr_path = data_path / 'expr_pca_obj.pickle'
print('Saving expr PCA object to', pca_expr_path)
with open(pca_expr_path, 'wb') as f:
    pickle.dump(pca_expr, f)

pca_expr_components_csv_path = data_path / 'pca_expr_components.csv'
print('Saving PCA expr components to', pca_expr_components_csv_path)
pca_expr_components.T.to_csv(pca_expr_components_csv_path)

pca_expr_components_pickle_path = replace_extension(pca_expr_components_csv_path, 'pickle')
print('Saving PCA expr components to', pca_expr_components_pickle_path)
pca_expr_components.T.to_csv(pca_expr_components_pickle_path)

with pd.HDFStore(find_newest_data_path('parse_tcga_mutations') / 'mutations.hdf5') as store:
    mutations = store['muts']

tx_feature_path = find_newest_data_path(f'treatment_features_alpha_{args.alpha:.2f}')

with pd.HDFStore(str(tx_feature_path / 'min_scores_full.hdf5')) as store:
    min_means_full = store['mean']
    min_stds_full = store['std']
    min_corrs_full = store['corr']

lincs_corr_raw = pd.read_pickle(find_newest_data_path('tcga_lincs_expr_features') / 'data.pickle')
lincs_corr_full = lincs_corr_raw.loc[min_stds_full.index, :].fillna(0)

lincs_corr_full.columns = [f'lincs_{drug}' for drug in lincs_corr_full.columns]

# In function so can garbage collect afterward
def load_full_matrices_and_run_pca(hdf5_path: Path, desc: str):
    print('Loading full feature matrices from', hdf5_path)
    pca_feature_matrices = {}
    with pd.HDFStore(str(hdf5_path)) as store:
        matrices_by_drug = {}
        for drug_with_slash in store:
            drug = drug_with_slash.lstrip('/')
            matrices_by_drug[drug] = store[drug_with_slash]

    pca_by_drug = {}
    components_by_drug = {}
    for drug, feature_matrix in matrices_by_drug.items():
        pca = PCA(n_components=PCA_COMPONENTS)
        feature_pca = pd.DataFrame(
            pca.fit_transform(feature_matrix),
            index=feature_matrix.index,
            columns=[f'{drug}_pca_{i}' for i in range(PCA_COMPONENTS)],
        )
        pca_by_drug[drug] = pca

        pca_components = pd.DataFrame(
            pca.components_,
            index=prop_muts_pca.columns,
            columns=prop_muts.columns,
        )
        components_by_drug[drug] = pca_components

        if args.plot_pca_components:
            plot_pca_component_values(pca_components, f'{drug}_{desc}')
            plot_pca_component_clustering(pca_components, f'{drug}_{desc}')
            plot_pca_component_subnetwork_cc(pca_components, f'{drug}_{desc}')

        pca_feature_matrices[drug] = feature_pca

    component_data_path = append_to_filename(data_path / hdf5_path.name, '_components')
    print('Saving PCA components by drug to', component_data_path)
    with pd.HDFStore(str(component_data_path)) as store:
        for drug, components in components_by_drug.items():
            store[drug] = components

    pca_obj_data_path = append_to_filename(replace_extension(data_path / hdf5_path.name, 'pickle'), '_pca')
    with open(pca_obj_data_path, 'wb') as f:
        pickle.dump(pca_by_drug, f)

    return pca_feature_matrices

mut_full_pca = load_full_matrices_and_run_pca(tx_feature_path / 'full_scores_mut.hdf5', 'muts')
expr_full_pca = load_full_matrices_and_run_pca(tx_feature_path / 'full_scores_diffexpr.hdf5', 'expr')

clinical_feature_cols = [
    'pathologic_T',
    'pathologic_N',
    'pathologic_M',
    'pathologic_stage',
    'histological_type',
    'icd_10_type',
    'icd_o_3_histology',
    'her2',
    'margin_status',
    'er_cell_percentage',
    #'breast_carcinoma_estrogen_receptor_status',
]
clinical_feature_df = tcga_clinical_data.loc[:, clinical_feature_cols].replace(no_value_pattern, value=nan)
for col in clinical_feature_cols:
    clinical_feature_df.loc[:, col] = clinical_feature_df.loc[:, col].astype('category')

clinical_feature_path = data_path / 'clinical_features.pickle'
print('Saving clinical features to', clinical_feature_path)
clinical_feature_df.to_pickle(clinical_feature_path)

print('Expanding clinical features into indicator columns')
clinical_feature_indicator_df = pd.get_dummies(clinical_feature_df)

dfs_to_consolidate = [
    (min_means_full, 'drug_mut_min_mean'),
    (min_stds_full, 'drug_mut_min_std'),
    (min_corrs_full, 'drug_mut_min_corr'),
    (prop_muts_pca, None),
    (prop_expr_pca, None),
    (lincs_corr_full, None),
    (clinical_feature_indicator_df, 'clinical'),
]
for drug in sorted(mut_full_pca):
    pca_feature_matrix = mut_full_pca[drug]
    dfs_to_consolidate.append((pca_feature_matrix, 'drug_mut_full'))
for drug in sorted(expr_full_pca):
    pca_feature_matrix = expr_full_pca[drug]
    dfs_to_consolidate.append((pca_feature_matrix, 'drug_expr_full'))

full_matrix_unscaled = consolidate_data_frames(dfs_to_consolidate).fillna(0)

full_matrix_unscaled_path = data_path / 'feature_matrix_unscaled.pickle'
print('Saving full matrix (unscaled) to', full_matrix_unscaled_path)
full_matrix_unscaled.to_pickle(full_matrix_unscaled_path)

data_desc_filepath = data_path / 'data_desc_unscaled.csv'
print('Saving unscaled data description to', data_desc_filepath)
full_matrix_unscaled.describe().T.to_csv(data_desc_filepath)

common_samples = sorted_intersection(full_matrix_unscaled.index, tx_info_raw.index)

scaler, full_matrix = scale_continuous_df_cols(full_matrix_unscaled)

full_matrix_csv_path = data_path / 'feature_matrix.csv'
print('Saving feature matrix to', full_matrix_csv_path)
full_matrix.to_csv(full_matrix_csv_path)

full_matrix_pickle_path = replace_extension(full_matrix_csv_path, 'pickle')
print('Saving feature matrix to', full_matrix_pickle_path)
full_matrix.to_csv(full_matrix_pickle_path)

data_desc_filepath = data_path / 'data_desc_normalized.csv'
print('Saving normalized data description to', data_desc_filepath)
full_matrix.describe().T.to_csv(data_desc_filepath)

data_matrix = full_matrix.loc[common_samples, :]
tx_info = tx_info_raw.loc[common_samples, :]

print('Treatment info column sums:')
print(tx_info.sum())

survival_data = pd.read_csv(drug_response_dir / 'survival.csv')
dead_samples = survival_data.index[survival_data.status == 1]

selected_drugs = ['arimidex', 'tamoxifen']

for drug in selected_drugs:
    # Patients on this drug, in each set
    drug_sample_sel = sorted_intersection(
        tx_info.index[tx_info.loc[:, f'rx_{drug}'] == 1],
        patient_selection,
    )
    switch_col_name = f'switch_from_{drug}'
    switched_from_drug_or_died = tx_info.loc[drug_sample_sel, switch_col_name].copy()
    drug_dead = sorted_intersection(drug_sample_sel, dead_samples)
    switched_from_drug_or_died.loc[drug_dead] = 1

    drug_feature_matrix_unscaled = full_matrix_unscaled.loc[drug_sample_sel, :]

    print('Shape of feature matrix for AI resistance:', drug_feature_matrix_unscaled.shape)

    drug_feature_matrix_path = data_path / f'feature_matrix_unscaled_{drug}.pickle'
    print(f'Saving unscaled AI drug feature matrix to {drug_feature_matrix_path}')
    drug_feature_matrix_unscaled.to_pickle(drug_feature_matrix_path)

    drug_feature_matrix = full_matrix.loc[drug_sample_sel, :]

    drug_feature_matrix_path = data_path / f'feature_matrix_{drug}.pickle'
    print(f'Saving AI drug feature matrix to {drug_feature_matrix_path}')
    drug_feature_matrix.to_pickle(drug_feature_matrix_path)

    label_path = data_path / f'labels_{drug}.pickle'
    print(f'Saving labels for {drug} to {label_path}')
    switched_from_drug_or_died.to_pickle(label_path)

# Bit of a hack. Compute labels and features for *all* patients for aromatase
# inhibitor response prediction; artificially mark patients as non-responsive
# if they weren't given the drug.
ai_drugs = ['arimidex', 'aromasin', 'femara']

pre_menopause_status = (
    'Pre (<6 months since LMP AND no prior bilateral '
    'ovariectomy AND not on estrogen replacement)'
)

rx_data = tx_info.loc[:, [f'rx_{drug}' for drug in ai_drugs]]
rx_any = (rx_data.sum(axis=1) > 0).astype(float)
ai_pred_sample_sel = (
    (tcga_clinical_data.er_status == 'Negative') |
    (rx_any == 1)
)
ai_pred_samples = sorted(ai_pred_sample_sel.index[ai_pred_sample_sel])

drug_samples = sorted_intersection(
    tx_info.index[rx_any == 1],
    ai_pred_samples,
)

switch_data = tx_info.loc[:, [f'switch_from_{drug}' for drug in ai_drugs]]
switch_any = (switch_data.sum(axis=1) > 0).astype(float)
non_response = pd.Series(0.0, index=ai_pred_samples)
other_samples = list(set(ai_pred_samples) - set(drug_samples))

er_negative_samples = sorted_intersection(
    tcga_clinical_data.index[
        tcga_clinical_data.er_status == 'Negative'
    ],
    ai_pred_samples,
)
print('ER- samples in selection:', len(er_negative_samples))

# Set "non-response" for all ER- samples, then set actual response
# values for patients who were given Arimidex (this order could matter if
# these sets weren't disjoint, but they are as per the set difference above)
non_response.loc[er_negative_samples] = 1

print(f'Assigning real response data for {len(drug_sample_sel)} samples')
non_response.loc[drug_samples] = switch_any.loc[drug_samples]

drug_feature_matrix_unscaled = full_matrix_unscaled.loc[ai_pred_samples, :]

print('Shape of feature matrix for AI resistance:', drug_feature_matrix_unscaled.shape)

drug_feature_matrix_path = data_path / 'feature_matrix_unscaled_ai_all.pickle'
print(f'Saving unscaled AI drug feature matrix to {drug_feature_matrix_path}')
drug_feature_matrix_unscaled.to_pickle(drug_feature_matrix_path)

drug_feature_matrix = full_matrix.loc[ai_pred_samples, :]

drug_feature_matrix_path = data_path / 'feature_matrix_ai_all.pickle'
print(f'Saving AI drug feature matrix to {drug_feature_matrix_path}')
drug_feature_matrix.to_pickle(drug_feature_matrix_path)

label_path = data_path / 'labels_ai_all.pickle'
print(f'Saving AI labels for to {label_path}')
non_response.to_pickle(label_path)
