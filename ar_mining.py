'''
Calculate support and confidence of rules
'''
import pandas, copy, operator
from dataclasses import dataclass
from common import RETAINED_DATA_COLUMNS, RETAINED_DATA_COLUMNS, MIN_SUP, MIN_CONF, RULE_ITEM, DATA_TUPLE

ORIGIN_DATA_FILE_PATH = 'dataset/adult-min-10.data'
MODIFIED_DATA_FILE_PATH = 'modified_ds.data'


OPERATORS = {
    '>=': operator.ge,
    '>': operator.gt,
    '<=': operator.le,
    '<': operator.lt,
}

@dataclass
class RULE:
    A: list
    B: list
    support_p: float
    support: int
    lhs_support: int
    confidence: float


# INPUT_RULES = [RULE([RULE_ITEM('*', 'sex')], [RULE_ITEM('*', 'race')], support_p=0.0, support=0, lhs_support=0, confidence=0.0)]
INPUT_RULES = [RULE([RULE_ITEM('(30, 50]', 'age')], [RULE_ITEM('*', 'marital-status')], support_p=0.0, support=0, lhs_support=0, confidence=0.0)]


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
    'native_country': [
        [None, '*'],
    ],
    'race': [
        [None, '*'],
    ],
    'sex': [
        [None, '*'],
    ],
    'education': [
        [None, '*'],
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


def data_tuple_supports_item_sets(rule_items: list, data_tuple: DATA_TUPLE):
    rule_items_check = [] 
    for rule_item in rule_items:
        tuple_value = data_tuple.get(rule_item.attr)
        if rule_item.value == tuple_value:
            rule_items_check.append(True)
        else:
            if type(tuple_value) in [float, int]:   # Handle numerical attributes
                # Construct number value range
                op_comparison_1 = '>=' if str(rule_item.value)[0] == '[' else '>'
                op_comparison_2 = '<=' if str(rule_item.value)[-1] == ']' else '<'
                l, h = rule_item.value[1: -1].split(',')
                l, h = float(l.strip()), float(h.strip())            
                return OPERATORS[op_comparison_1](tuple_value, l) and OPERATORS[op_comparison_2](tuple_value, h) 
            if type(tuple_value) is str:   # Handle categorical attributes
                # Check tuple value in a tree branch in which the rule item value is one of ancestors
                for tree_branch in CAT_TREES[rule_item.attr]:                
                    if tree_branch[0] is None or (tuple_value == tree_branch[0] and rule_item.value in tree_branch):
                        rule_items_check.append(True)
                return rule_items_check.append(False)

        print('FAILED', tuple_value, rule_item.attr, data_tuple)
        raise Exception('WHAT"S THE FUCK!')

    return all(rule_items_check)

def cal_supp_conf(data_file_path, columns, input_rules):
    rules = copy.deepcopy(input_rules)
    D = pandas.read_csv(data_file_path, names=columns,
                    index_col=False, skipinitialspace=True)
    dataset_length = D.shape[0]
    print('Dataset length', dataset_length)
    for _, data_tuple in D.iterrows():
        print('Data tuple', data_tuple.get('age'))
        for rule in rules:
            # Check if this data_tuple supports this rule A->B
            data_tuple_supports_rule_lhs_A = data_tuple_supports_item_sets(rule.A, data_tuple)
            data_tuple_supports_rule = data_tuple_supports_item_sets(rule.A + rule.B, data_tuple)
            if data_tuple_supports_rule:
                rule.support += 1
            if data_tuple_supports_rule_lhs_A:
                rule.lhs_support += 1

    for rule in rules:
        rule.support_p = rule.support / dataset_length
        rule.confidence = rule.support / rule.lhs_support
        
    return rules


rules_from_origin_ds = cal_supp_conf(ORIGIN_DATA_FILE_PATH, RETAINED_DATA_COLUMNS, INPUT_RULES)
print('RESULTS')
print('RULES FROM ORIGIN DATASET')
print(rules_from_origin_ds)
# rules_from_modified_ds = cal_supp_conf(MODIFIED_DATA_FILE_PATH, RETAINED_DATA_COLUMNS, INPUT_RULES)
# print('RULES FROM MODIFIED DATASET')
# print(rules_from_modified_ds)
