##################
# Import libraries
##################

import pandas as pd
import numpy as np
import math
import itertools
import json


###########
# Load data
###########

DATA_COLUMNS = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
RETAINED_DATA_COLUMNS = ['age', 'sex', 'marital-status', 'native-country',
                         'race', 'education', 'hours-per-week', 'capital-gain', 'workclass']

adult = pd.read_csv('dataset/adult.data', names=DATA_COLUMNS, index_col=False, skipinitialspace=True)
adult = adult[RETAINED_DATA_COLUMNS]
print(adult)

################
# Pre-processing
################

#drop redundant column
# adult = adult.drop("education-num", axis=1)

#deal with ? entries
columns_with_blank_entries = ["workclass", "native-country"]
for column in columns_with_blank_entries:
    adult["converted_" +
          column] = adult[column].astype(str).replace(" ", "")+"_"+column
    adult = adult.drop(column, axis=1)

#deal with numeric attribures
numeric_columns = ["age", "capital-gain", "hours-per-week"]
for column in numeric_columns:
    bins = np.histogram(adult[column])
    bins = list(bins[1])  # generate 10 equal-width bins
    if(bins[0] == 0.0):
        bins[0] = -0.1
    category = pd.cut(adult[column], bins)
    category = category.to_frame()
    category.columns = ['converted_'+column]
    adult = pd.concat([adult, category], axis=1)
    adult["converted_"+column] = column+"_" + \
        adult["converted_"+column].astype(str).replace(" ", "")
    adult = adult.drop(column, axis=1)

#convert to dictionary format for using as an input to the program
dict_table = {}
temp_list = []
for index, data in adult.iterrows():
    dict_table[str(index)] = data.tolist()


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
    for x in range(len(values)):
        temp_C.append({"itemset": [values[x]], "support": counts[x]})
    return temp_C


def apriori(dict_table, support):
    L = []
    C = []
    min_sup_count = len(dict_table)*support

    #generate frequent-1-itemsets. L[0] holds the 1-itemsets, while C[0] holds the 1-itemsets with their support counts
    print("Generating Frequent 1-itemsets")
    # generates candidates for 1-itemsets and their counts
    counts_C1 = generate_counts_C1(dict_table)
    pruned_L1, pruned_C1 = prune(
        counts_C1, min_sup_count)  # pruning against min_sup
    L.append(pruned_L1)
    for x in pruned_C1:
        x['support'] = (float(x['support'])/float(len(dict_table)))
    C.append(pruned_C1)

    k = 2
    while(1):
        print("Generating Frequent " + str(k) + "-itemsets")
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

def rule_confidence(left_hand_side, right_hand_side):
    pass


if __name__ == '__main__':    
    support = 0.3
    L, C = apriori(dict_table, support=support)
    number_of_frequent_itemsets = sum(len(x) for x in L)
    print(" ")
    print("Number of frequent itemsets:")
    print(number_of_frequent_itemsets)
    print(" ")
    print("Frequent itemsets with support: ")
    print(json.dumps(C, indent=4))
