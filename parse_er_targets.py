#!/usr/bin/env python3
import csv
from pathlib import Path

from data_path_utils import create_data_path

data_path = create_data_path('parse_er_targets')

trrust_path = Path('~/data/trrust_rawdata.txt').expanduser()

er_names = {'ESR1', 'ESR2'}
er_targets = set()

with trrust_path.open() as f:
    r = csv.reader(f, delimiter='\t')
    for line in r:
        tf_name = line[0]
        target_name = line[1]
        if tf_name in er_names:
            er_targets.add(target_name)

target_path = data_path / 'er_targets.txt'
print('Saving {} ER targets to {}'.format(len(er_targets), target_path))
with target_path.open('w') as f:
    for target in sorted(er_targets):
        print(target, file=f)
