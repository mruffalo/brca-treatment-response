#!/usr/bin/env python3
from functools import partial
from pathlib import Path

import attr
from data_path_utils import pathlib_walk_glob
import lxml.etree

CLINICAL_XML_PATTERN = '*clinical*.xml'

@attr.s
class Drug:
    name = attr.ib(init=False, default='')
    response = attr.ib(init=False, default='')

@attr.s
class Patient:
    path = attr.ib()

    disease = attr.ib(init=False, default='')
    patient_id = attr.ib(init=False, default='')
    patient_uuid = attr.ib(init=False, default='')
    survival_time = attr.ib(init=False, default=0)
    dead = attr.ib(init=False, default=False)
    drugs = attr.ib(init=False, default=attr.Factory(list))

    pathologic_T = attr.ib(init=False, default=None)
    pathologic_N = attr.ib(init=False, default=None)
    pathologic_M = attr.ib(init=False, default=None)
    pathologic_stage = attr.ib(init=False, default=None)
    histological_type = attr.ib(init=False, default=None)
    icd_10_type = attr.ib(init=False, default=None)
    icd_O_3_histology = attr.ib(init=False, default=None)
    her2 = attr.ib(init=False, default=None)
    margin_status = attr.ib(init=False, default=None)
    er_status = attr.ib(init=False, default=None)
    er_cell_percentage = attr.ib(init=False, default=None)
    menopause_status = attr.ib(init=False, default=None)

    def assign_value_if_possible(self, attr_name, root, nsmap, xpath):
        try:
            value = root.xpath(xpath, namespaces=nsmap)[0]
        except Exception:
            # Don't care whether it's an exception from lxml or an IndexError
            # if there's no text in this node; just leave as default value
            return

        setattr(self, attr_name, value)

# List of 2-tuples:
#  [0] Patient attribute name
#  [1] xpath tag name
clinical_fields = [
    ('pathologic_T', 'shared_stage:pathologic_T'),
    ('pathologic_N', 'shared_stage:pathologic_N'),
    ('pathologic_M', 'shared_stage:pathologic_M'),
    ('pathologic_stage', 'shared_stage:pathologic_stage'),
    ('histological_type', 'shared:histological_type'),
    ('icd_10_type', 'clin_shared:icd_10'),
    ('icd_O_3_histology', 'clin_shared:icd_o_3_histology'),
    ('her2', 'brca_shared:lab_proc_her2_neu_immunohistochemistry_receptor_status'),
    ('margin_status', 'clin_shared:margin_status'),
    ('er_status', 'brca_shared:breast_carcinoma_estrogen_receptor_status'),
    ('er_cell_percentage', 'brca_shared:er_level_cell_percentage_category'),
    ('menopause_status', 'clin_shared:menopause_status'),
]

def parse_clinical_xml(xml_file: Path) -> Patient:
    """
    :param xml_file: TCGA clincal XML file
    :return: Patient object
    """
    patient = Patient(xml_file)

    print('Parsing', xml_file)
    with xml_file.open('rb') as f:
        tree = lxml.etree.parse(f)

    root = tree.getroot()
    nsmap = root.nsmap.copy()

    disease_xpath = '//admin:disease_code/text()'
    patient.disease = root.xpath(disease_xpath, namespaces=nsmap)[0].lower()

    patient_id_xpath = '//shared:bcr_patient_barcode/text()'
    patient.patient_id = root.xpath(patient_id_xpath, namespaces=nsmap)[0]

    patient_uuid_xpath = '//shared:bcr_patient_uuid/text()'
    patient.patient_uuid = root.xpath(patient_uuid_xpath, namespaces=nsmap)[0].lower()

    for attr_name, tag_name in clinical_fields:
        patient.assign_value_if_possible(attr_name, root, nsmap, f'//{tag_name}/text()')

    vital_status_xpath = '//clin_shared:vital_status/text()'
    try:
        patient.dead = root.xpath(vital_status_xpath, namespaces=nsmap)[0] == 'Dead'
    except IndexError:
        # No survival status; leave blank
        pass

    time_tag_name = 'days_to_death' if patient.dead else 'days_to_last_followup'
    time_xpath = '//clin_shared:{}/text()'.format(time_tag_name)
    try:
        patient.survival_time = float(root.xpath(time_xpath, namespaces=nsmap)[0])
    except IndexError:
        # No survival time; leave blank
        pass

    drug_node_xpath = '//rx:drugs/rx:drug'
    drug_name_xpath = 'rx:drug_name/text()'
    drug_response_xpath = 'clin_shared:measure_of_response/text()'
    drug_nodes = root.xpath(drug_node_xpath, namespaces=nsmap)
    for node in drug_nodes:
        drug = Drug()

        name_results = node.xpath(drug_name_xpath, namespaces=nsmap)
        if name_results:
            drug.name = name_results[0]

        response_results = node.xpath(drug_response_xpath, namespaces=nsmap)
        if response_results:
            drug.response = response_results[0]

        patient.drugs.append(drug)

    return patient

find_clinical_xml_files = partial(pathlib_walk_glob, pattern=CLINICAL_XML_PATTERN)
