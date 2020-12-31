import csv
import pandas
import math
import random
import time
import traceback
import sys
import copy
import pickle
from dataclasses import dataclass
from itertools import combinations
from common import *
from eval import eval_results


OUTPUT_DATASET_PATH = 'modified_ds.data'


def budgets_of_rules_affected_by_migrating_a_tuple(R_care, t: DATA_TUPLE, group_j: GROUP):
    '''Calculate budgets of rules affected by migrating a data tuple to another group'''
    R_affected = construct_r_affected_by_a_migration(R_care, [t], group_j)
    return [rule.budget for rule in R_affected]


# Migrate the data tuple into the most useful group of a safe group
def find_group_to_move_dispersing(R_care: list, ug_data_tuples: list, SG: list, random_choice=False):
    min_no_rule_budgets = 99999
    affected_rules = []
    result_group = None
    if random_choice:
        result_group = random.choice(SG)
        print('Random choose group {} for dispersing small group'.format(result_group.index))
        R_affected = construct_r_affected_by_a_migration(R_care, ug_data_tuples, result_group)
    else:
        for considering_dst_group in SG:
            R_affected = construct_r_affected_by_a_migration(R_care, ug_data_tuples, considering_dst_group)
            no_rules_with_negative_budget = sum(1 for rule in R_affected if rule.budget < 0)

            if no_rules_with_negative_budget < min_no_rule_budgets:
                result_group = considering_dst_group
                min_no_rule_budgets = no_rules_with_negative_budget
                affected_rules = R_affected

    return result_group, affected_rules
                        

def cal_number_of_free_tuples(groups: list, k: int):
    if not k:
        K_SET = list(range(5, 30, 5))
    else:
        K_SET = [k]
    for k in K_SET:
        no_free_tuples = 0
        for group in groups:
            no_free_tuples += group_length(group) - k if group_length(group) >= k else 0
        print('NUMBER OF FREE TUPLES: {}'.format(no_free_tuples))


def generate_free_tuples(SG: list):
    '''
    Pick free tuples in each safe group to establish a set of free tuples. Just pick the neccessary tuples at first of each safe group
    '''
    free_tuples = []
    for group in SG:
        group_free_tuples = group.origin_tuples[:group_length(group) - DESIRED_K]
        free_tuples.extend(group_free_tuples)

    return free_tuples


def cal_budgets_threshold(rules: list, THRESHOLD=0.8):
    '''
    Calculate a threshold for rule budgets to perform loops 
    Try to preserve a percentage equals to <THRESHOLD> of rules
    '''
    rules.sort(key=lambda rule: rule, reverse=True)


def all_rules_having_acceptable_budgets(rules: list):
    return all([rule.budget > 0 for rule in rules])


def all_rules_having_positive_budgets(rules: list):
    return all([rule.budget > 0 for rule in rules])


def m3ar_modified_algo(D, R_initial, output_file_name='m3ar_modified.data'):    
    start_time = time.time()
    # First construct R care
    R_care = construct_r_care(R_initial)
    for r in R_care:
        r.budget = rule_budget(r)
    print('R CARE LENGTH =', len(R_care))
    # pprint_rule_set(R_care)
    rule_budgets = [rule.budget for rule in R_care]
    print('R CARE BUDGETS AT INITIAL:')
    print(rule_budgets)

    print('===============================================================================')    
    # Build groups from the dataset then split G into 2 sets of groups: safe groups SG and unsafe groups UG
    # Sort groups in UG and SG by length ascendingly
    GROUPS, SG, UG = build_groups(D)
    print('K =', DESIRED_K)
    print('THERE ARE {} SAFE GROUPS AND {} UNSAFE GROUPS'.format(len(SG), len(UG)))
    cal_number_of_free_tuples(GROUPS, DESIRED_K)
    free_tuples = generate_free_tuples(SG)    
    SelG = None
    loop_iteration = 0
    # STAGE 1: Pick free tuples into unsafe groups
    while len(UG) > 0 and len(free_tuples) > 0 and all_rules_having_positive_budgets(R_care):
        loop_iteration += 1
        if SelG is None:
            # MODIFICATION 1: UG was already sorted by group length. Process the UG groups one by one, start with the unsafe group with the longest length
            # The longer the group length, the more chance it will soon become a safe group
            SelG = UG.pop(0)
            print('LOOP ITERATION {}. Pop the unsafe group with the longest length. SelG index: {}. SelG length: {}'.format(loop_iteration, SelG.index, group_length(SelG)))
            # print('Rules budget now are:')
            # print([rule.budget for rule in R_care])

            no_tuples_needed_to_become_a_safe_group = DESIRED_K - group_length(SelG)
            no_tuples_picked = 0
            free_tuples_to_remove = []
            for data_tuple in free_tuples:
                R_affected = construct_r_affected_by_a_migration(R_care, [data_tuple], SelG)
                if all_rules_having_positive_budgets(R_affected):
                    for rule in R_affected:
                        rule.budget -= 1
                    # Perform a migration of this free tuple to the destination group                           
                    convert_quasi_attributes(data_tuple, SelG)
                    SelG.received_tuples.append(data_tuple)
                    print('Data tuple picked is from group {}'.format(data_tuple.group_index))
                    source_group = find_group(data_tuple.group_index, GROUPS)
                    source_group.origin_tuples.remove(data_tuple)
                    # free_tuples.remove(data_tuple)
                    free_tuples_to_remove.append(data_tuple)
                    no_tuples_picked += 1
                    if no_tuples_picked == no_tuples_needed_to_become_a_safe_group:
                        # Pick enough, break to process the next unsafe group                        
                        add_group(SelG, SG)
                        remove_group(SelG, UG)
                        print('Number of safe groups now is: {}. Number of free tuples is: {}'.format(len(SG), len(free_tuples)))
                        SelG = None
                        break

            for t in free_tuples_to_remove:
                free_tuples.remove(t)

            # If look up in all free tuples but cannot find enough tuples to make this group safe, return them to source group
            if no_tuples_picked < no_tuples_needed_to_become_a_safe_group:                
                while len(SelG.received_tuples) > 0:
                    data_tuple = SelG.received_tuples[0]
                    SelG.received_tuples.remove(data_tuple)
                    source_group = find_group(data_tuple.group_index, GROUPS)                    
                    convert_quasi_attributes(data_tuple, source_group)
                    source_group.origin_tuples.append(data_tuple)
                    R_affected = construct_r_affected_by_a_migration(R_care, [data_tuple], source_group)
                    free_tuples.append(data_tuple)
                    add_group(SelG, UG)
                    remove_group(SelG, SG)
                    for rule in R_affected:
                        rule.budget += 1

                SelG = None

    print('TOTAL LOOPS: {}'.format(loop_iteration))
    print('AFTER STAGE 1')
    print('TOTAL NUMBER OF SAFE GROUPS: {}'.format(len([group for group in GROUPS if group_length(group) >= DESIRED_K])))
    print('TOTAL NUMBER OF UNSAFE GROUPS: {}'.format(len([group for group in GROUPS if 0 < group_length(group) < DESIRED_K])))
    print('NUMBER OF WRONG SAFE GROUPS: {}'.format(len([g for g in SG if group_length(g) < DESIRED_K])))
    print('NUMBER OF UNSAFE GROUPS AND SAFE GROUPS: {}, {}'.format(len(UG), len(SG)))
    print('TOTAL NUMBER OF TUPLES IN SAFE GROUPS: {}'.format(sum(group_length(group) for group in GROUPS if group_length(group) >= DESIRED_K)))
    print('TOTAL NUMBER OF TUPLES IN UNSAFE GROUPS: {}'.format(sum(group_length(group) for group in GROUPS if group_length(group) < DESIRED_K)))
    print('NUMBER OF FREE TUPLES: {}'.format(len(free_tuples)))
    rule_budgets = [rule.budget for rule in R_care]
    print('R CARE BUDGETS AFTER STAGE 1:')
    print(rule_budgets)

    # STAGE 2: PROCESS ONE BY ONE IN THE UNSAFE GROUP, START WITH GROUP WITH SHORT LENGTH
    UG = sorted(UG, key=lambda gr: group_length(gr))
    UG_SMALL_DISPERSED, UG_BIG_DISPERSED = [], []
    for unsafe_group in UG:
        if group_length(unsafe_group) <= DESIRED_K / 2:  # With small group disperse them
            st = time.time()
            dst_group, R_affected = find_group_to_move_dispersing(R_care, unsafe_group.origin_tuples, SG, random_choice=True)
            print('Find group to move dispersing step takes', time.time() - st, 'seconds', 'Number of safe groups:', len(SG))
            if dst_group:
                for rule in R_affected:
                    rule.budget -= 1                
                print('DISPERSE SMALL GROUP', unsafe_group.index, '- LENGTH BEFORE:', group_length(unsafe_group))
                for data_tuple in unsafe_group.origin_tuples:
                    convert_quasi_attributes(data_tuple, dst_group)
                    dst_group.received_tuples.append(data_tuple)                                    
                for data_tuple in unsafe_group.received_tuples:
                    convert_quasi_attributes(data_tuple, dst_group)
                    dst_group.received_tuples.append(data_tuple)
                unsafe_group.origin_tuples = []
                unsafe_group.received_tuples = []
                print('DISPERSE SMALL GROUP', unsafe_group.index, '- LENGTH AFTER:', group_length(unsafe_group))
                # remove_group(unsafe_group, UG)
                UG_SMALL_DISPERSED.append(unsafe_group)
            else:
                print('Cannot find any group for group {} to disperse to'.format(unsafe_group.index))
        else:
            no_tuples_needed_to_become_a_safe_group = DESIRED_K - group_length(unsafe_group)
            picked_tuples = free_tuples[:no_tuples_needed_to_become_a_safe_group]
            if len(picked_tuples) < no_tuples_needed_to_become_a_safe_group:    # This means the free tuples set is now empty
                break
            # Migrate these free tuples to this unsafe group to make it safe
            R_affected = construct_r_affected_by_a_migration(R_care, picked_tuples, unsafe_group)
            for rule in R_affected:
                rule.budget -= 1
            for data_tuple in picked_tuples:
                convert_quasi_attributes(data_tuple, unsafe_group)
            print('UNSAFE BIG GROUP {} RECEIVED {} FREE TUPLES TO BECOME SAFE'.format(unsafe_group.index, no_tuples_needed_to_become_a_safe_group))
            unsafe_group.received_tuples.extend(picked_tuples)
            add_group(unsafe_group, SG) # Now becomes safe
            for data_tuple in picked_tuples:
                source_group = find_group(data_tuple.group_index, GROUPS)
                print('SAFE GROUP {} GIVES TUPLES TO UNSAFE BIG GROUP {}'.format(source_group.index, unsafe_group.index))
                source_group.origin_tuples.remove(data_tuple)
                free_tuples.remove(data_tuple)
            # remove_group(unsafe_group, UG)
            UG_BIG_DISPERSED.append(unsafe_group)

    for g_sm in UG_SMALL_DISPERSED:
        print('Small group dispersed', g_sm.index, group_length(g_sm))
        remove_group(g_sm, UG)

    for g_big in UG_BIG_DISPERSED:
        print('Big group dispersed', g_big.index, group_length(g_big))
        remove_group(g_big, UG)

    print('AFTER STAGE 2')
    print('NUMBER OF UNSAFE GROUPS AND SAFE GROUPS: {}, {}'.format(len([ug for ug in UG if group_length(ug) > 0]), len(SG)))
    print('NUMBER OF UNSAFE GROUPS WITH LENGTH <= K/2 DISPERSED (MOVED TO A SAFE GROUP): {}'.format(len(UG_SMALL_DISPERSED)))
    print('NUMBER OF UNSAFE GROUPS WITH LENGTH > K/2 DISPERSED (RECEIVED FREE TUPLES): {}'.format(len(UG_BIG_DISPERSED)))
    print('TOTAL NUMBER OF SAFE GROUPS: {}'.format(len([group for group in GROUPS if group_length(group) >= DESIRED_K])))
    print('TOTAL NUMBER OF UNSAFE GROUPS: {}'.format(len([group for group in GROUPS if 0 < group_length(group) < DESIRED_K])))
    print('TOTAL NUMBER OF TUPLES IN SAFE GROUPS: {}'.format(sum(group_length(group) for group in GROUPS if group_length(group) >= DESIRED_K)))
    print('TOTAL NUMBER OF TUPLES IN UNSAFE GROUPS: {}'.format(sum(group_length(group) for group in GROUPS if group_length(group) < DESIRED_K)))
    print('NUMBER OF FREE TUPLES: {}'.format(len(free_tuples)))
    print('UG_SMALL_DISPERSED', [g.index for g in UG_SMALL_DISPERSED])
    print('UG_BIG_DISPERSED', [g.index for g in UG_BIG_DISPERSED])
    rule_budgets = [rule.budget for rule in R_care]
    print('R CARE BUDGETS:')
    print(rule_budgets)

    # STAGE 3: PROCESS ONE BY ONE IN THE REMAINING UNSAFE GROUP, START WITH GROUP WITH SHORT LENGTH
    for unsafe_group in UG:
        # first_tuple_of_this_unsafe_group = unsafe_group.origin_tuples[0]
        dst_group, R_affected = find_group_to_move_dispersing(R_care, unsafe_group.origin_tuples, SG)
        if dst_group:
            for rule in R_affected:                
                rule.budget -= 1
            for data_tuple in unsafe_group.origin_tuples:
                convert_quasi_attributes(data_tuple, dst_group)
                dst_group.received_tuples.append(data_tuple)                                    
            for data_tuple in unsafe_group.received_tuples:
                convert_quasi_attributes(data_tuple, dst_group)
                dst_group.received_tuples.append(data_tuple)
            unsafe_group.origin_tuples = []
            unsafe_group.received_tuples = []
        else:
            print('Cannot find any group for group {} to disperse to'.format(unsafe_group.index))

    print('TOTAL NUMBER OF TUPLES IN SAFE GROUPS: {}'.format(sum(group_length(group) for group in GROUPS if group_length(group) >= DESIRED_K)))
    # print('TOTAL NUMBER OF TUPLES IN SAFE GROUPS 2: {}'.format(sum(group_length(group) for group in SG)))
    print('TOTAL NUMBER OF TUPLES IN UNSAFE GROUPS: {}'.format(sum(group_length(group) for group in GROUPS if group_length(group) < DESIRED_K)))
    # print('TOTAL NUMBER OF TUPLES IN UNSAFE GROUPS 2: {}'.format(sum(group_length(group) for group in UG)))

    eval_results(R_initial, GROUPS, output_file_name, start_time)


if __name__ == '__main__':
    # sys.stdout = open("log/modified_algo_results.log", "w")
    if len(sys.argv) > 3:
        data_file_path, initial_rules_path, DESIRED_K = sys.argv[1], sys.argv[2], int(sys.argv[3])
    else:
        data_file_path = 'dataset/adult-prep.data'
        initial_rules_path = 'adult-prep-rules-picked.data'
    # A dataset reaches k-anonymity if total risks of all groups equals to 0
    # A Member Migration operation g(i)-T-g(j) is valuable when the risk of data is decreased after performing that Member Migration operation.
    D = pandas.read_csv(data_file_path, names=RETAINED_DATA_COLUMNS, index_col=False, skipinitialspace=True)
    dataset_length = D.shape[0]
    print('DATASET LENGTH=', dataset_length)
    print('MIN_SUP=', MIN_SUP)
    print('MIN_CONF=', MIN_SUP)
    print('K=', DESIRED_K)
    MIN_SUP = MIN_SUP * dataset_length
    R_initial = []
    with open(initial_rules_path, 'rb') as f:
        R_initial = pickle.load(f)
    # print('R initial', R_initial)
    output_file_name = 'out_mod_m3ar_' + str(DESIRED_K) + '_' + data_file_path.split('/')[-1].split('.')[0] + '.data'
    m3ar_modified_algo(D, R_initial, output_file_name)

    sys.stdout.close()
