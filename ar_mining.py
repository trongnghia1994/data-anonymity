'''
Calculate support and confidence of rules
'''
import pandas, copy    
from dataclasses import dataclass

ORIGIN_DATA_FILE_PATH = 'dataset/adult-min-100.data'
MODIFIED_DATA_FILE_PATH = 'modified_ds.data'

@dataclass
class RULE:
    A: list
    B: list
    support_p: float
    support: int
    lhs_support: int
    confidence: float


@dataclass
class RULE_ITEM:
    value: any
    attr: str

DATA_COLUMNS = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
RETAINED_DATA_COLUMNS = ['age', 'sex', 'marital-status', 'native-country',
                         'race', 'education', 'hours-per-week', 'capital-gain', 'workclass']
MIN_SUP = 0.3

INPUT_RULES = [RULE([RULE_ITEM('Male', 'sex')], [RULE_ITEM('White', 'race')], support_p=0.0, support=0, lhs_support=0, confidence=0.0)]

def cal_supp_conf(data_file_path, columns, input_rules):
    rules = copy.deepcopy(input_rules)
    D = pandas.read_csv(data_file_path, names=columns,
                    index_col=False, skipinitialspace=True)
    dataset_length = D.shape[0]
    print('Dataset length', dataset_length)
    for index, data_tuple in D.iterrows():
        for rule in rules:
            lhs_support, rule_support = 0, 0
            # Check if this data_tuple supports this rule A->B
            data_tuple_supports_rule_lhs_A = all(data_tuple.get(rule_item.attr) == rule_item.value for rule_item in rule.A)
            data_tuple_supports_rule = all(data_tuple.get(rule_item.attr) == rule_item.value for rule_item in (rule.A + rule.B))
            if data_tuple_supports_rule:
                rule.support += 1
            if data_tuple_supports_rule_lhs_A:
                rule.lhs_support += 1

    for rule in rules:
        rule.support_p = rule.support / dataset_length
        rule.confidence = rule.support / rule.lhs_support
        
    return rules


rules_from_origin_ds = cal_supp_conf(ORIGIN_DATA_FILE_PATH, DATA_COLUMNS, INPUT_RULES)
rules_from_modified_ds = cal_supp_conf(MODIFIED_DATA_FILE_PATH, RETAINED_DATA_COLUMNS, INPUT_RULES)
print('RESULTS')
print(rules_from_origin_ds)
print(rules_from_modified_ds)
