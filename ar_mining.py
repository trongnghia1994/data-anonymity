'''
Calculate support and confidence of rules
'''
import pandas, copy, operator
from dataclasses import dataclass
from common import *

ORIGIN_DATA_FILE_PATH = 'dataset/adult-min-10.data'
MODIFIED_DATA_FILE_PATH = 'modified_ds.data'


# INPUT_RULES = [RULE([RULE_ITEM('*', 'sex')], [RULE_ITEM('*', 'race')], support_p=0.0, support=0, lhs_support=0, confidence=0.0)]


def cal_supp_conf(data_file_path, columns, input_rules):
    for rule in input_rules:
        rule.support = 0
        rule.confidence = 0
        rule.lhs_support = 0
        rule.support_p = 0.0
        rule.budget = 0

    rules = copy.deepcopy(input_rules)
    D = pandas.read_csv(data_file_path, names=columns,
                    index_col=False, skipinitialspace=True)
    dataset_length = D.shape[0]
    for _, data_tuple in D.iterrows():
        for rule in rules:
            # Check if this data_tuple supports this rule A->B
            data_tuple_supports_rule_lhs_A = data_tuple_supports_item_sets(rule.A, data_tuple)
            all_rules_item = rule.A + rule.B            
            data_tuple_supports_rule = data_tuple_supports_item_sets(all_rules_item, data_tuple)            
            if data_tuple_supports_rule:
                rule.support += 1
            if data_tuple_supports_rule_lhs_A:
                rule.lhs_support += 1

    for rule in rules:
        rule.support_p = rule.support / dataset_length
        try:
            rule.confidence = rule.support / rule.lhs_support
        except:            
            rule.confidence = 0
        
    return rules


if __name__ == '__main__':
    INPUT_RULES = [
        RULE(A=[RULE_ITEM(value='(-0.1-2523.6]', attr='capital-gain'), RULE_ITEM(value='HS-grad', attr='education'), RULE_ITEM(value='(30.4-40.2]', attr='hours-per-week'), RULE_ITEM(value='Married-civ-spouse', attr='marital-status'), RULE_ITEM(value='White', attr='race')], B=[RULE_ITEM(value='United-States', attr='native-country'), RULE_ITEM(value='Male', attr='sex'), RULE_ITEM(value='Private', attr='workclass')], lhs_support=0, support_p=0.0, support=36, confidence=0.5538461538461539, budget=0),
        RULE(A=[RULE_ITEM(value='HS-grad', attr='education'), RULE_ITEM(value='(30.4-40.2]', attr='hours-per-week'), RULE_ITEM(value='Married-civ-spouse', attr='marital-status'), RULE_ITEM(value='Male', attr='sex')], B=[RULE_ITEM(value='United-States', attr='native-country'), RULE_ITEM(value='White', attr='race'), RULE_ITEM(value='(-0.1-2523.6]', attr='capital-gain'), RULE_ITEM(value='Private', attr='workclass')], lhs_support=0, support_p=0.0, support=36, confidence=0.5, budget=0),
        RULE(A=[RULE_ITEM(value='HS-grad', attr='education'), RULE_ITEM(value='(30.4-40.2]', attr='hours-per-week'), RULE_ITEM(value='Married-civ-spouse', attr='marital-status'), RULE_ITEM(value='Private', attr='workclass')], B=[RULE_ITEM(value='United-States', attr='native-country'), RULE_ITEM(value='White', attr='race'), RULE_ITEM(value='Male', attr='sex'), RULE_ITEM(value='(-0.1-2523.6]', attr='capital-gain')], lhs_support=0, support_p=0.0, support=36, confidence=0.5806451612903226, budget=0),
    ]
    # rules = cal_supp_conf('output/out_m3ar_k_5_adult-min-1000-prep.data', RETAINED_DATA_COLUMNS, INPUT_RULES)
    rules = cal_supp_conf('output/out_m3ar_k_5_adult-min-1000-prep.data', RETAINED_DATA_COLUMNS, INPUT_RULES)
    print('RESULTS')
    print('RULES')
    for rule in rules:
        pprint_rule(rule)
