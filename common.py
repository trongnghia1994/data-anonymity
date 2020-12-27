from dataclasses import dataclass
import pandas
import pickle
import random
import operator


DATA_FILE_PATH = './dataset/adult-prep.data'
DATA_COLUMNS = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
RETAINED_DATA_COLUMNS = ['age', 'sex', 'marital-status', 'native-country',
                         'race', 'education', 'hours-per-week', 'capital-gain', 'workclass']
QUASI_ATTRIBUTES = RETAINED_DATA_COLUMNS[:6]

MIN_SUP = 0.03
MIN_CONF = 0.5
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
    lhs_support: int = 0
    support_p: float = 0.0
    support: int = 0
    confidence: float = 0.0
    budget: float = 0.0


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
    quasi_attributes_values: list


@dataclass
class DATA_TUPLE:
    index: int
    data: pandas.Series
    group_index: int


def is_safe_group(a_group: GROUP):
    return group_length(a_group) >= DESIRED_K


def is_unsafe_group(a_group: GROUP):
    return not is_safe_group(a_group)


def group_length(a_group: GROUP):
    return (len(a_group.origin_tuples) + len(a_group.received_tuples))


def build_groups(dataset: pandas.DataFrame, quasi_attrs: list = QUASI_ATTRIBUTES):
    '''Build safe groups and unsafe groups from the initial dataset'''
    UG, SG = [], []
    DF_GROUPS = dataset.groupby(quasi_attrs)
    group_index = 0
    for _, df_group in DF_GROUPS:
        group_data = []
        for row in df_group.iterrows():
            index, data = row
            data_tuple = DATA_TUPLE(index, data, group_index)
            group_data.append(data_tuple)

        group = GROUP(group_index, len(group_data), group_data, [], group_data[0].data.values)
        if is_safe_group(group):
            SG.append(group)
        else:
            UG.append(group)

        group_index += 1

    GROUPS = SG + UG
    return GROUPS, SG, UG


def group_first_tuple(a_group: GROUP):
    '''Return the first tuple of group from origin_tuples or received_tuples'''
    if len(a_group.origin_tuples) > 0:
        return a_group.origin_tuples[0]

    if len(a_group.received_tuples) > 0:
        return a_group.origin_tuples[0]

    return None


def convert_quasi_attributes(data_tuple: DATA_TUPLE, dst_group: GROUP):
    '''Convert all quasi attributes' values of <data_tuple> 
    to the corresponding of <dst_data_tuple>'''
    if group_length(dst_group) > 0:
        dst_data_tuple = group_first_tuple(dst_group)
        data_tuple.data.update(dst_data_tuple.data[:len(QUASI_ATTRIBUTES)])
    else:
        data_tuple.data.update(dst_group.quasi_attributes_values)        


def pprint_data_tuple(data_tuple: DATA_TUPLE):
    str_concat = '{}: '.format(data_tuple.index)
    for index, value in data_tuple.data.items():
        str_concat += str(value)
        str_concat += ','
    print(str_concat)


def pprint_groups(groups: list):
    for group in groups:
        print('================================')
        print('Group index', group.index)
        print('Group length:', group_length(group), '===== Is safe?', is_safe_group(group))
        print('Group origin tuples:', 'Empty' if len(group.origin_tuples) == 0 else '')
        for t in group.origin_tuples:
            pprint_data_tuple(t)
        print('Group received tuples:', 'Empty' if len(group.received_tuples) == 0 else '')
        for t in group.received_tuples:
            pprint_data_tuple(t)
        print('================================')


def export_dataset(groups: list, output_file_name: str):
    '''Write the modified dataset to file'''
    def write_data_tuple(t: DATA_TUPLE, f):
        str_concat = ''
        for index, value in t.data.items():
            str_concat += str(value)
            str_concat += ','

        str_concat = str_concat[:-1]
        f.write(str_concat + '\n')
        
    with open(output_file_name, 'w') as f:
        for group in groups:
            for t in group.origin_tuples:
                write_data_tuple(t, f)

            for t in group.received_tuples:
                write_data_tuple(t, f)


# METRICS FOR DATA QUALITY
def newly_generated_rules_percentage(D):
    pass


def no_loss_rules(D):
    pass


def different_rules_percentage(D):
    pass


def metrics_cavg(groups: list):
    return sum(group_length(group) for group in groups) / len(groups)


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


def read_rules_data(data_file_path='initial_rules.data'):
    # Read list of rules from binary pickled file
    with open(data_file_path, 'rb') as f:
        data = pickle.load(f)
        return data

    return None


def pick_random_rules(no_rules: int, data_file_path='initial_rules.data'):
    rules = read_rules_data(data_file_path)
    random.sample(rules, no_rules)
    return random.sample(rules, no_rules)


def pprint_rule(rule: RULE):
    print('({}) => ({}) - Support: {}, Confidence: {}, Budget: {}, LHS Support: {}'.format(','.join([rule_item.value for rule_item in rule.A]), ','.join([rule_item.value for rule_item in rule.B]), rule.support, rule.confidence, rule.budget, rule.lhs_support))
    

def pprint_rule_set(rules: list):
    for rule in rules:
        pprint_rule(rule)


CAT_TREES = {
    'marital-status': [
        ['Never-married', '*'],
        ['Married-civ-spouse', 'Married', '*'],
        ['Married-AF-spouse', 'Married', '*'],
        ['Divorced', 'leave', '*'],
        ['Separated', 'leave', '*'],
        ['Widowed', 'alone', '*'],
        ['Married-spouse-absent', 'alone', '*'],
    ],
    'native-country': [
        ['Cambodia','*'],
        ['Canada','*'],
        ['China','*'],
        ['Columbia','*'],
        ['Cuba','*'],
        ['Dominican-Republic','*'],
        ['Ecuador','*'],
        ['El-Salvador','*'],
        ['England','*'],
        ['France','*'],
        ['Germany','*'],
        ['Greece','*'],
        ['Guatemala','*'],
        ['Haiti','*'],
        ['Holand-Netherlands','*'],
        ['Honduras','*'],
        ['Hong','*'],
        ['Hungary','*'],
        ['India','*'],
        ['Iran','*'],
        ['Ireland','*'],
        ['Italy','*'],
        ['Jamaica','*'],
        ['Japan','*'],
        ['Laos','*'],
        ['Mexico','*'],
        ['Nicaragua','*'],
        ['Outlying-US(Guam-USVI-etc)','*'],
        ['Peru','*'],
        ['Philippines','*'],
        ['Poland','*'],
        ['Portugal','*'],
        ['Puerto-Rico','*'],
        ['Scotland','*'],
        ['South','*'],
        ['Taiwan','*'],
        ['Thailand','*'],
        ['Trinadad&Tobago','*'],
        ['United-States','*'],
        ['Vietnam','*'],
        ['Yugoslavia','*'],
    ],
    'race': [
        ['Amer-Indian-Eskimo','*'],
        ['Asian-Pac-Islander','*'],
        ['Black','*'],
        ['Other','*'],
        ['White','*'],
    ],
    'sex': [
        ['Female','*'],
        ['Male','*'],
    ],
    'education': [
        ['10th','*'],
        ['11th','*'],
        ['12th','*'],
        ['1st-4th','*'],
        ['5th-6th','*'],
        ['7th-8th','*'],
        ['9th','*'],
        ['Assoc-acdm','*'],
        ['Assoc-voc','*'],
        ['Bachelors','*'],
        ['Doctorate','*'],
        ['HS-grad','*'],
        ['Masters','*'],
        ['Preschool','*'],
        ['Prof-school','*'],
        ['Some-college','*'],
    ],
    'workclass': [
        ['Private', '*'],
        ['Self-emp-not-inc', 'Self-employ', '*'],
        ['Self-emp-inc', 'Self-employ', '*'],
        ['Federal-gov', 'gov', '*'],
        ['Local-gov', 'gov', '*'],
        ['State-gov', 'gov', '*'],
        ['Without-pay', 'not-work', '*'],
        ['Never-worked', 'not-work', '*'],
    ],
}


OPERATORS = {
    '>=': operator.ge,
    '>': operator.gt,
    '<=': operator.le,
    '<': operator.lt,
}


def data_tuple_supports_item_sets(rule_items: list, data_tuple: DATA_TUPLE):
    for rule_item in rule_items:
        step_res = True
        tuple_value = data_tuple.get(rule_item.attr)
        if rule_item.value == tuple_value:
            step_res = True
        else:   # Handle Generalization values
            # Handle numerical attributes age 1    rule_item:age (20, 30])
            if type(tuple_value) in [float, int] and type(rule_item.value) is str:
                # Construct number value range
                op_comparison_1 = '>=' if str(rule_item.value)[0] == '[' else '>'
                op_comparison_2 = '<=' if str(rule_item.value)[-1] == ']' else '<'
                l, h = rule_item.value[1: -1].split(',')
                l, h = float(l.strip()), float(h.strip())       
                step_res = OPERATORS[op_comparison_1](tuple_value, l) and OPERATORS[op_comparison_2](tuple_value, h)
            elif type(tuple_value) is str:   # Handle categorical attributes
                # Check tuple value in a tree branch in which the rule item value is one of ancestors
                is_in_cat_tree = False
                for tree_branch in CAT_TREES[rule_item.attr]:
                    if tuple_value == tree_branch[0] and rule_item.value in tree_branch[1:]:                        
                        is_in_cat_tree = True
                        break
                step_res = is_in_cat_tree
            else:
                step_res = False
        if not step_res:
            return False
    return True


def data_tuple_supports_a_rule(data_tuple: DATA_TUPLE, rule: RULE):
    return data_tuple_supports_item_sets((rule.A + rule.B), data_tuple)
