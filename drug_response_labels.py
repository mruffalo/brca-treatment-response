#!/usr/bin/env python3
from collections import Counter, defaultdict
from itertools import chain
import json
from math import nan
from pprint import pprint
import re
from typing import Mapping, Set

from data_path_utils import (
    DATA_PATH,
    create_data_path,
    find_newest_data_path,
)
import pandas as pd

SWITCH_COL_PATTERN = 'switch_from_{}'
RX_COL_PATTERN = 'rx_{}'

# TODO: figure out a better way to do this
SWITCH_COL_RE = re.compile(SWITCH_COL_PATTERN.replace('{}', r'(\w+)'))
RX_COL_RE = re.compile(RX_COL_PATTERN.replace('{}', r'(\w+)'))

data_path = create_data_path('drug_response_labels')

# will change to 'luad' soon
selected_cancer = 'brca'
# Change for lung cancer
drugs = {'arimidex', 'aromasin', 'femara', 'tamoxifen'}

json_input_path = find_newest_data_path('tcga_xml_to_json')
with (json_input_path / 'tcga_clinical_data.json').open() as f:
    clinical_data = json.load(f)

clinical_data_selected_cancer = []
for patient_data in clinical_data:
    if patient_data['disease'] == selected_cancer:
        clinical_data_selected_cancer.append(patient_data)

clinical_data_selected_cancer_df = pd.DataFrame(clinical_data_selected_cancer).drop('drugs', axis=1)
clinical_data_selected_cancer_path = data_path / f'clinical_data_{selected_cancer}.pickle'
print('Saving clinical data DataFrame to', clinical_data_selected_cancer_path)
clinical_data_selected_cancer_df.to_pickle(clinical_data_selected_cancer_path)

drug_counts = Counter()
for patient_data in clinical_data_selected_cancer:
    drug_counts.update(drug['name'].lower() for drug in patient_data['drugs'])

print('Drug counts:')
pprint(drug_counts)

# Update synonyms as appropriate, with misspellings and brand vs. generic, etc.
# This is an easy way to represent this for people to read -- the key maps to
# a list of synonyms, which can be easily modified later
synonyms = {
    'arimidex': [
        'anastrazole',
        'anastrozole',
        'anastrozole (arimidex)',
        'arimidex',
        'arimidex (anastrozole)',
    ],
    'tamoxifen': [
        'nolvadex',
        'tamoxifen',
        'tamoxifen (novadex)',
    ],
    'aromasin': [
        'exemestane',
        'aromasin (exemestane)',
        'aromasin',
        'exemestane (aromasin)',
        'aromatase exemestane',
    ],
    'femara': [
        'femara (letrozole)',
        'letrozole (femara)',
        'femara',
        'letrozole',
    ],
}

# ... but for actual use in this script, we want to invert the mapping, so that
# every element of the above lists maps to the "real" name
synonym_mapping = {}
for primary_name, extra_names in synonyms.items():
    for name in extra_names:
        synonym_mapping[name] = primary_name

patients_on_drug = defaultdict(set)
patients_switch_from_drug = defaultdict(set)

all_samples_set = set(patient['patient_id'] for patient in clinical_data_selected_cancer)

# TODO: refactor this as appropriate
# Read UPMC clinical information, if available, and merge this with
# TCGA patient information
upmc_tx_info_path = DATA_PATH / 'upmc_treatment_info_{}.csv'.format(selected_cancer)
if upmc_tx_info_path.is_file():
    upmc_tx_info = pd.read_csv(upmc_tx_info_path, index_col=0)
    all_samples_set.update(upmc_tx_info.index)
    for col in upmc_tx_info.columns:
        selected_patients = set(upmc_tx_info.index[upmc_tx_info.loc[:, col] == 1])

        m = RX_COL_RE.match(col)
        if m:
            drug_name = m.group(1)
            patients_on_drug[drug_name].update(selected_patients)

        m = SWITCH_COL_RE.match(col)
        if m:
            drug_name = m.group(1)
            patients_switch_from_drug[drug_name].update(selected_patients)

all_samples = sorted(all_samples_set)

survival_time = pd.Series(nan, index=all_samples)
# 0: alive, 1: dead
survival_status = pd.Series(nan, index=all_samples)

for patient_data in clinical_data_selected_cancer:
    patient_id = patient_data['patient_id']
    if patient_data['dead']:
        survival_status.loc[patient_id] = 1
    else:
        survival_status.loc[patient_id] = 0

    survival_time.loc[patient_id] = patient_data['survival_time']

print('{} samples have no survival time'.format(survival_time.isnull().sum()))
print('{} samples have no survival status'.format(survival_status.isnull().sum()))

survival_df = pd.DataFrame({'survival': survival_time, 'status': survival_status})
survival_path = data_path / 'survival.csv'
print('Saving survival data to', survival_path)
survival_df.to_csv(survival_path)

# Assume a patient switched from the drug if they had any of these responses
bad_response_values = {
    'Clinical Progressive Disease',
    'Stable Disease',
}

for patient_data in clinical_data_selected_cancer:
    patient_id = patient_data['patient_id']
    for drug in patient_data['drugs']:
        drug_name = drug['name'].lower()
        adjusted_name = synonym_mapping.get(drug_name, drug_name)

        if adjusted_name not in drugs:
            # Skip; not interested in this one
            continue

        patients_on_drug[adjusted_name].add(patient_id)
        if drug['response'] in bad_response_values:
            patients_switch_from_drug[adjusted_name].add(patient_id)

# Set non-response to 1 for all patients who died, also
dead_samples = set(survival_status.index[survival_status == 1])
for drug in drugs:
    dead_samples_on_drug = dead_samples & patients_on_drug[drug]
    patients_switch_from_drug[drug].update(dead_samples_on_drug)

def patients_by_drug_to_pairs(patients_by_drug: Mapping[str, Set[str]], col_pattern: str):
    for drug, patients in patients_by_drug.items():
        col = col_pattern.format(drug)
        for patient in patients:
            yield (patient, col)

# Not using pairs_to_zero_one_dataframe, since we need all columns and all
# patients, regardless of whether a patient was actually prescribed anything

columns = sorted(
    chain(
        [RX_COL_PATTERN.format(drug) for drug in drugs],
        [SWITCH_COL_PATTERN.format(drug) for drug in drugs],
    )
)

tx_info = pd.DataFrame(0.0, index=all_samples, columns=columns)

for drug in drugs:
    tx_info.loc[patients_on_drug[drug], RX_COL_PATTERN.format(drug)] = 1
    tx_info.loc[patients_switch_from_drug[drug], SWITCH_COL_PATTERN.format(drug)] = 1

print(tx_info.sum())

tx_info_path = data_path / 'tx_info.pickle'
print('Saving treatment info to', tx_info_path)
tx_info.to_pickle(tx_info_path)
