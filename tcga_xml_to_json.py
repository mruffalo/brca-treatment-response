#!/usr/bin/env python3
import json
from pathlib import Path
import pickle

import attr
from data_path_utils import create_data_path
import pandas as pd

from parse_tcga_clinical_xml import Patient, find_clinical_xml_files, parse_clinical_xml

data_path = create_data_path('tcga_xml_to_json')

input_path = Path('~/data/tcga-clinical-all').expanduser()
print(f'Parsing XML files in {input_path}')
patients = [parse_clinical_xml(path) for path in find_clinical_xml_files(input_path)]
print(f'Read information for {len(patients)} patients')
converted_data = [
    attr.asdict(
        patient,
        filter=attr.filters.exclude(attr.fields(Patient).path),
    )
    for patient in patients
]

pickle_file = data_path / 'tcga_clinical_data.pickle'
print('Saving pickled data to', pickle_file)
with open(pickle_file, 'wb') as f:
    pickle.dump(patients, f)

output_file = data_path / 'tcga_clinical_data.json'
print('Saving JSON data to', output_file)
with open(output_file, 'w') as f:
    json.dump(converted_data, f)

converted_data_no_drugs = [
    attr.asdict(
        patient,
        filter=attr.filters.exclude(
            attr.fields(Patient).path,
            attr.fields(Patient).drugs,
        ),
    )
    for patient in patients
]

df_for_csv = pd.DataFrame(converted_data_no_drugs)
csv_file = data_path / 'tcga_clinical_data.csv'
print('Saving CSV data (without drugs) to', csv_file)
df_for_csv.to_csv(csv_file, index=False)
