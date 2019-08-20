#!/usr/bin/env python3
import re

# TODO: use version in utils after fixing circular import and caring about this minor code duplication
def strip_suffix(string: str, suffix: str) -> str:
    """
    :param string: String to strip `suffix` from, if present
    :param suffix: Suffix to remove from `string`
    :return:
    """
    if string.endswith(suffix):
        return string[:-len(suffix)]
    return string

test_data = '''
arimidex_expr_drug_mut_min_mean
prop_expr_pca_0
er_only_pca_0_drug_expr_full
er_targets_expr_drug_mut_min_mean
er_targets_pca_0_drug_mut_full
aromasin_pca_0_drug_expr_full
aromasin_expr_drug_mut_min_mean
er_only_expr_drug_mut_min_mean
aromasin_pca_3_drug_expr_full
er_only_expr_drug_mut_min_std
arimidex_pca_7_drug_expr_full
er_targets_pca_3_drug_expr_full
er_targets_mut_drug_mut_min_mean
aromasin_expr_drug_mut_min_std
aromasin_pca_9_drug_mut_full
aromasin_mut_drug_mut_min_mean
aromasin_pca_6_drug_expr_full
femara_expr_drug_mut_min_mean
er_targets_pca_7_drug_expr_full
er_targets_mut_drug_mut_min_std
histological_type_Infiltrating Lobular Carcinoma_clinical
arimidex_pca_1_drug_expr_full
femara_pca_3_drug_expr_full
er_only_pca_1_drug_expr_full
er_only_pca_3_drug_mut_full
arimidex_pca_0_drug_expr_full
er_only_pca_9_drug_mut_full
aromasin_pca_8_drug_mut_full
er_targets_pca_5_drug_expr_full
lincs_taxol_corr
lincs_taxol_dot_product
er_only_pca_1_drug_mut_full
er_cell_percentage_90-99%_clinical
femara_pca_7_drug_expr_full
femara_pca_0_drug_expr_full
er_only_pca_2_drug_expr_full
prop_expr_pca_2
prop_expr_pca_1
er_targets_pca_2_drug_expr_full
er_targets_pca_0_drug_expr_full
'''

min_mean_std_pattern = re.compile(r'([\w_]+)_(expr|mut)_drug_mut_min_(mean|std|corr)')
plain_pca_pattern = re.compile(r'prop_(expr|mut)_pca_(\d+)')
min_pca_pattern = re.compile(r'([\w_]+)_pca_(\d+)_drug_(expr|mut)_full')
lincs_corr_pattern = re.compile(r'lincs_(\w+)_(corr|dot_product)')

clinical_suffix = '_clinical'

# Could make this smarter, but eh
drug_mapping = {
    'arimidex': 'Arimidex targets',
    'aromasin': 'Aromasin targets',
    'er_only': '{ESR1, ESR2}',
    'er_targets': 'ER targets',
    'femara': 'Femara targets',
}

data_type_mapping = {
    'expr': 'diff. expr',
    'mut': 'som. muts',
}

lincs_op_mapping = {
    'corr': 'Correlation',
    'dot_product': 'Dot product',
}

def convert_label(label: str) -> str:
    """
    >>> convert_label('arimidex_expr_drug_mut_min_mean')
    'Min. of [sm. Arimidex targets, sm. diff. expr], mean across genes'
    >>> convert_label('prop_expr_pca_0')
    'Smoothed diff. expr, PCA component 0'
    >>> convert_label('er_only_pca_0_drug_expr_full')
    'Min. of [sm. {ESR1, ESR2}, sm. diff. expr], PCA component 0'
    >>> convert_label('lincs_taxol_dot_product')
    'Dot product: sample expr. vs. LINCS cell line given Taxol'
    >>> convert_label('histological_type_Infiltrating Lobular Carcinoma_clinical')
    'histological_type_Infiltrating Lobular Carcinoma'

    >>> convert_label('')
    ''
    """
    m = min_mean_std_pattern.match(label)
    if m:
        drug_desc = drug_mapping[m.group(1)]
        data_type_desc = data_type_mapping[m.group(2)]
        mean_or_std = m.group(3)
        return f'Min. of [sm. {drug_desc}, sm. {data_type_desc}], {mean_or_std} across genes'

    m = plain_pca_pattern.match(label)
    if m:
        data_type_desc = data_type_mapping[m.group(1)]
        pca_component = m.group(2)
        return f'Smoothed {data_type_desc}, PCA component {pca_component}'

    m = min_pca_pattern.match(label)
    if m:
        drug_desc = drug_mapping[m.group(1)]
        pca_component = m.group(2)
        data_type_desc = data_type_mapping[m.group(3)]
        return f'Min. of [sm. {drug_desc}, sm. {data_type_desc}], PCA component {pca_component}'

    m = lincs_corr_pattern.match(label)
    if m:
        drug = m.group(1).title()
        op_desc = lincs_op_mapping[m.group(2)]
        return f'{op_desc}: sample expr. vs. LINCS cell line given {drug}'

    if label.endswith(clinical_suffix):
        return strip_suffix(label, clinical_suffix)

    # fallback
    return label

if __name__ == '__main__':
    for label in test_data.strip().splitlines():
        print(convert_label(label))
