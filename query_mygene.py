#!/usr/bin/env python3
import json
from typing import Iterable

from data_path_utils import (
    create_data_path,
    find_newest_data_path,
)
import mygene
import pandas as pd

def get_genes() -> Iterable[str]:
    hit_data_dir = find_newest_data_path('tf_mirna_hits_both')
    hits = pd.read_pickle(hit_data_dir / 'hits_max_annotated_in_transmir.pickle')
    tfs = set(tfn.split('::')[0] for tfn in hits.tf_name)
    return tfs

genes = get_genes()

mg = mygene.MyGeneInfo()

q = mg.querymany(
    genes,
    species='human',
    scopes=['ensemblgene', 'entrezgene', 'symbol'],
    fields=['ensembl.gene', 'entrezgene', 'symbol'],
)

data_path = create_data_path('query_mygene')

raw_results_path = data_path / 'raw_results.json'
print('Saving raw query results to', raw_results_path)
with open(raw_results_path, 'w') as f:
    json.dump(q, f)

mapping = {}
for result in q:
    if 'entrezgene' in result:
        mapping[result['query']] = str(result['entrezgene'])

mapping_path = data_path / 'mapping.json'
print('Saving mapping to', mapping_path)
with open(mapping_path, 'w') as f:
    json.dump(mapping, f)

entrez_hugo_mapping = {}
for result in q:
    if 'entrezgene' in result and 'symbol' in result:
        entrez_hugo_mapping[result['entrezgene']] = result['symbol']

print(
    'Mapped {} Entrez IDs to {} HUGO symbols'.format(
        len(entrez_hugo_mapping),
        len(set(entrez_hugo_mapping.values())),
    )
)

entrez_hugo_mapping_path = data_path / 'entrez_hugo_mapping.json'
print('Saving Entrez -> HUGO mapping to', entrez_hugo_mapping_path)
with open(entrez_hugo_mapping_path, 'w') as f:
    json.dump(entrez_hugo_mapping, f)
