"""Preprocess the input dataset and generate the rules file. Make sure file paths are all true"""
from common import pick_random_rules, pprint_rule_set, QUASI_ATTRIBUTES, RETAINED_DATA_COLUMNS, metrics_cavg, pprint_rule, metrics_cavg_raw, rules_metrics
from ar_mining import cal_supp_conf
from preprocess import preprocess
from Apriori import apriori_gen_rules
import pickle
import subprocess
import time
import pandas

# Below is the input dataset, may cut the origin file to make it smaller for testing
# input_ds = 'dataset/adult.data'
input_ds = 'dataset/adult-min-100.data'

# Preprocess the dataset: discard null/unknown values
processed_ds = preprocess(input_ds)
print('DONE Preprocessing. Preprocessed file path:', processed_ds)

# Generate the set of rules (into a file)
rules_data_file, R_initial = apriori_gen_rules(processed_ds)
print('DONE generate ruels. Length of R_initial:', len(R_initial))
# Pick some initial rules then write them into a file
# R_initial = pick_random_rules(10, rules_data_file)

# Check the picked rules file at adult-prep-rules-picked.data
picked_rules_file = rules_data_file.split('.')[0] + '-picked.data'
# pprint_rule_set(R_initial)
with open(picked_rules_file, 'wb') as f:
    pickle.dump(R_initial, f)

print('DONE dump rules to file:', picked_rules_file)
