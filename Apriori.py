##################
# Import libraries
##################

import pandas as pd
import numpy as np
import math
import itertools
import json
import pickle
from itertools import combinations
from common import DATA_FILE_PATH, RETAINED_DATA_COLUMNS, RULE, RULE_ITEM, MIN_SUP, MIN_CONF, gen_rule_hash_value, QUASI_ATTRIBUTES
numeric_columns = ["capital-gain", "hours-per-week"]


###########
# Functions
###########

def generate_counts_C(candidate_C, dict_table):
    temp_C = []
    for candidate_itemset in candidate_C:
        temp_C.append({"itemset": candidate_itemset, "support": 0})
        for dict_list in dict_table.values():
            if(all(x in dict_list for x in candidate_itemset)):
                temp_C[-1]["support"] += 1
    return temp_C


def generate_counts_itemset(candidate_itemset, dict_table):
    temp_C = 0
    for dict_list in dict_table.values():
        if(all(x in dict_list for x in candidate_itemset)):
            temp_C += 1
    return temp_C


# make sure all subsets are frequent
def has_infrequent_subset(candidate_itemset, set_itemset, k):
    subsets = list(itertools.combinations(candidate_itemset, k-1))
    for x in subsets:
        if(list(x) not in set_itemset):
            return True
    return False


# make sure the first k-2 elements are same
def check_share_condition(itemset_l1, itemset_l2, num_share):
    for x in range(num_share):
        if(itemset_l1[x] != itemset_l2[x]):
            return False
    if(itemset_l1[num_share] >= itemset_l2[num_share]):  # violate -> false
        return False
    return True


def generate_candidate_C(set_itemset, k):
    num_share = k-2
    candidate_C = []
    for itemset_l1_index, itemset_l1 in enumerate(set_itemset):
        for itemset_l2_index, itemset_l2 in enumerate(set_itemset):
            # join members of L_{k-1} if their first k-2 items are common
            if(check_share_condition(itemset_l1, itemset_l2, num_share)):
                candidate_itemset = sorted(
                    list(set().union(itemset_l1, itemset_l2)))  # join step
                if(has_infrequent_subset(candidate_itemset, set_itemset, k) is False):  # prune step
                    candidate_C.append(candidate_itemset)
    return candidate_C


def prune(temp_C, min_sup_count):
    pruned_temp_C = [x for x in temp_C if x["support"] >= min_sup_count]
    pruned_temp_L = sorted([x["itemset"]
                            for x in temp_C if x["support"] >= min_sup_count])
    return pruned_temp_L, pruned_temp_C


def generate_counts_C1(dict_table):
    temp_C = []
    items = [item for dict_list in dict_table.values() for item in dict_list]
    values, counts = np.unique(items, return_counts=True)
    for i in range(len(values)):
        temp_C.append({"itemset": [values[i]], "support": counts[i]})
    return temp_C


def apriori(dict_table, support):
    L = []
    C = []
    min_sup_count = len(dict_table)*support

    #generate frequent-1-itemsets. L[0] holds the 1-itemsets, while C[0] holds the 1-itemsets with their support counts
    # print("Generating Frequent 1-itemsets")
    # generates candidates for 1-itemsets and their counts
    counts_C1 = generate_counts_C1(dict_table)
    pruned_L1, pruned_C1 = prune(counts_C1, min_sup_count)  # pruning against min_sup
    L.append(pruned_L1)
    for x in pruned_C1:
        x['support'] = int(x['support'])
        x['support_p'] = (float(x['support'])/float(len(dict_table)))
    C.append(pruned_C1)

    k = 2
    while(1):
        # print("Generating Frequent " + str(k) + "-itemsets")
        # same as the 'apriori_gen' procedure in the book
        candidate_C = generate_candidate_C(L[k-2], k=k)
        counts_C = generate_counts_C(candidate_C, dict_table)  # find counts
        # prune if support condition is not met
        pruned_L, pruned_C = prune(counts_C, min_sup_count)

        if(not pruned_L):  # break if pruned L is an empty set
            break

        L.append(pruned_L)
        for x in pruned_C:
            x['support_p'] = (float(x['support'])/float(len(dict_table)))
        C.append(pruned_C)
        k = k + 1
    return L, C


################
# Initialization
################

def gen_rules(item_set, dict_table, min_conf, output_file):
    result = []
    items = item_set['itemset']
    rule_index = 0
    for i in range(1, len(items)):
        lhs_candidates = list(itertools.combinations(items, i))
        for lhs in lhs_candidates:            
            rhs = tuple(set(items) - set(lhs))
            lhs_support = generate_counts_itemset(lhs, dict_table)
            rule_confidence = float(item_set['support']) / float(lhs_support)
            if rule_confidence >= min_conf:
                rule = RULE(index=rule_index, A=[], B=[], support=0, confidence=0, budget=0)
                # print(lhs, '===>', rhs, 'support=', item_set['support'], 'confidence=', rule_confidence)                    
                for rule_item in lhs:
                    attr, value = rule_item.split('_')                    
                    if attr in numeric_columns: # Replace the , with -
                        value = value.replace(' ', '')
                        comma_pos = value.rfind(',')
                        if comma_pos > -1:
                            value = value[:comma_pos] + '-' + value[comma_pos + 1:]
                    rule.A.append(RULE_ITEM(value, attr))
                    if attr in QUASI_ATTRIBUTES:
                        rule.quasi.add('{}_{}'.format(attr, value))

                for rule_item in rhs:
                    attr, value = rule_item.split('_')
                    if attr in numeric_columns: # Replace the , with -
                        value = value.replace(' ', '')
                        comma_pos = value.rfind(',')
                        if comma_pos > -1:
                            value = value[:comma_pos] + '-' + value[comma_pos + 1:]
                    rule.B.append(RULE_ITEM(value, attr))
                    if attr in QUASI_ATTRIBUTES:
                        rule.quasi.add('{}_{}'.format(attr, value))

                rule.support = item_set['support']
                rule.confidence = rule_confidence
                rule.hash_value = gen_rule_hash_value(rule)
                rule.index = rule_index
                result.append(rule)
                rule_index += 1

    return result


def apriori_gen_rules(input_ds=DATA_FILE_PATH):
    adult = pd.read_csv(input_ds, names=RETAINED_DATA_COLUMNS, index_col=False, skipinitialspace=True)
    
    # deal with numeric attributes
    for column in numeric_columns:
        bins = np.histogram(adult[column])
        bins = list(bins[1])  # generate 10 equal-width bins
        if(bins[0] == 0.0):
            bins[0] = -0.1
        category = pd.cut(adult[column], bins)
        category = category.to_frame()
        category.columns = ['converted_'+column]
        adult = pd.concat([adult, category], axis=1)
        adult["converted_"+column] = column + "_" + adult["converted_"+column].astype(str).replace(" ", "")
        adult = adult.drop(column, axis=1)


    # Process remaining columns
    for column in RETAINED_DATA_COLUMNS:
        if column not in numeric_columns:
            adult["converted_" + column] = column + "_" + adult[column].astype(str).replace(" ", "")
            adult = adult.drop(column, axis=1)


    #convert to dictionary format for using as an input to the program
    dict_table = {}
    temp_list = []
    for index, data in adult.iterrows():
        dict_table[str(index)] = data.tolist()
    
    output_file=input_ds.split('/')[1].split('.')[0] + '-rules.data'
    L, C = apriori(dict_table, support=MIN_SUP)
    number_of_frequent_itemsets = sum(len(x) for x in L)
    # print("Number of frequent itemsets:")
    # print(number_of_frequent_itemsets)
    # print("Frequent itemsets with support: ")
    # print(json.dumps(C, indent=4))
    # Process from the itemsets with length = 2
    all_rules = []
    for item_set_group in C[1:]:
        for item_set in item_set_group:
            all_rules.extend(gen_rules(item_set, dict_table, MIN_CONF, output_file))

    with open(output_file, 'wb') as f:
        pickle.dump(all_rules, f)
    
    return output_file, all_rules


if __name__ == '__main__':    
    apriori_gen_rules()
