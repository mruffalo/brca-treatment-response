from data_path_utils import find_newest_data_path

from gene_mappings import read_hugo_entrez_mapping

hugo_entrez_mapping = read_hugo_entrez_mapping()

# Manual queries to ChEMBL: https://www.ebi.ac.uk/chembl/
# (for each selected drug)
# Keyword search (on right side) for drug name
# Get "Target Predictions" for each; I used proteins/genes with score >= 0.5, add to
# 'targets_raw'

entrez_ids_manual = {
    'DDR1': '780',
    'DDR2': '4921',
    'CYP19A1': '1588',
    'PDE4A': '5141',
    'PGGT1B': '5229',
    'ESR1': '2099',
    'ESR2': '2100',
    'HRH1': '3269',
    'HRH2': '3274',
    'HRH3': '11255',
    'CYP26A1': '1592',
    'CYP11B1': '1584',
    'CYP11B2': '1585',
    'ARSC': '412',
    'SRD5A2': '6716',
    'SHBG': '6462',
    'SRD5A1': '6715',
    'HIF1A': '3091',
    'CYP17A1': '1586',
    'HSD17B1': '3292',
    'NFKB1': '4790',
    'HSD17B3': '3293',
    'BLM': '641',
    'NR3C1': '2908',
    'HSD11B2': '3291',
}

entrez_id_mapping = hugo_entrez_mapping.copy()
entrez_id_mapping.update(entrez_ids_manual)

targets_raw = {
    'er_only': [
        'ESR1',
        'ESR2',
    ],
    'arimidex': [
        'CYP19A1',
        'DDR1',
        'DDR2',
        'PDE4A',
        'PGGT1B',
    ],
    'aromasin': [
        'SRD5A2',
        'CYP19A1',
        'SHBG',
        'SRD5A1',
        'HIF1A',
        'CYP17A1',
        'HSD17B1',
        'ARSC',
        'NFKB1',
        'HSD17B3',
        'BLM',
        'NR3C1',
        'HSD11B2',
    ],
    'femara': [
        'ARSC',
        'CYP11B1',
        'CYP11B2',
        'CYP19A1',
        'CYP26A1',
    ],
}

er_target_path = find_newest_data_path('parse_er_targets') / 'er_targets.txt'
with er_target_path.open() as f:
    print('Assigning ER targets from', er_target_path)
    targets_raw['er_targets'] = set(line.strip() for line in f)

targets = {
    drug: [entrez_id_mapping[target] for target in target_list]
    for drug, target_list in targets_raw.items()
}

drugs = sorted(targets)
