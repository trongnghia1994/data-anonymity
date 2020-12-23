from dataclasses import dataclass
import pandas
import pickle

DATA_FILE_PATH = './dataset/adult.data'
DATA_COLUMNS = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
RETAINED_DATA_COLUMNS = ['age', 'sex', 'marital-status', 'native-country',
                         'race', 'education', 'hours-per-week', 'capital-gain', 'workclass']
QUASI_ATTRIBUTES = RETAINED_DATA_COLUMNS[:6]

DATA_FILE_PATH = './dataset/adult.data'
DATA_COLUMNS = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
RETAINED_DATA_COLUMNS = ['age', 'sex', 'marital-status', 'native-country',
                         'race', 'education', 'hours-per-week', 'capital-gain', 'workclass']
QUASI_ATTRIBUTES = RETAINED_DATA_COLUMNS[:6]
MIN_SUP = 0.03
MIN_CONF = 0.5
MIN_LENGTH = 2
DESIRED_K = 5

'''
    Notions
    Rule A->B
        a_rule:
            A
            B
            support
            confidence
        MIN_SUP sm
        MIN_CONF cm
        Support (A->B) = s
        Confidence (A->B) = c
    Group: Set of tuples
'''


@dataclass
class RULE:
    A: list
    B: list
    support: float
    confidence: float
    budget: float


@dataclass
class RULE_ITEM:
    value: any
    attr: str


@dataclass
class GROUP:
    index: int
    origin_len: int
    origin_tuples: list
    received_tuples: list


@dataclass
class DATA_TUPLE:
    index: int
    data: pandas.Series
    group_index: int


# METRICS FOR DATA QUALITY
def newly_generated_rules_percentage(D):
    pass


def no_loss_rules(D):
    pass


def different_rules_percentage(D):
    pass


def remove_group(group: GROUP, group_list: list):
    if group in group_list:
        group_list.remove(group)


def add_group(group: GROUP, group_list: list):
    if group not in group_list:
        group_list.append(group)


def find_group(group_index, groups: list = []):
    '''Find group by its index'''
    for group in groups:
        if group.index == group_index:
            return group

    return None


def read_rules_data(data_file_path='output_rules.log'):
    # Read list of rules from binary pickled file
    with open('output_rules.log', 'rb') as f:
        data = pickle.load(f)
        return data

    return None


__all__ = ['DATA_FILE_PATH', 'DATA_COLUMNS', 'RETAINED_DATA_COLUMNS', 'QUASI_ATTRIBUTES', 'MIN_SUP', 'MIN_CONF', 'DESIRED_K', 'RULE', 'RULE_ITEM', 'GROUP', 'DATA_TUPLE', 'add_group', 'remove_group', 'find_group', 'read_rules_data']
