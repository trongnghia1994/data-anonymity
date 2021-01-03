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
from common import *
from m3ar import find_group_to_migrate, disperse
from eval import eval_results


OUTPUT_DATASET_PATH = 'modified_ds.data'


# Migrate the data tuple into the most useful group of a safe group
def find_group_to_move_dispersing(ug: GROUP, SG: list, random_choice=False):
    min_no_rule_budgets = 99999
    affected_rules = []
    result_group = None
    if random_choice:
        result_group = random.choice(SG)
        print('Random choose group {} for dispersing small group'.format(result_group.index))
        R_affected = construct_r_affected_by_a_migration(ug, result_group)
    else:
        for considering_dst_group in SG:
            R_affected = construct_r_affected_by_a_migration(ug, considering_dst_group)
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


def generate_free_tuples(SG: list, k=DESIRED_K):
    '''
    Pick free tuples in each safe group to establish a set of free tuples. Just pick the neccessary tuples at first of each safe group
    '''
    free_tuples = []
    for group in SG:
        group_free_tuples = group.origin_tuples[:group_length(group) - k]
        free_tuples.extend(group_free_tuples)

    return free_tuples


def all_rules_having_acceptable_budgets(rules: list):
    return all([rule.budget > 0 for rule in rules])


def all_rules_having_positive_budgets(rules: list):
    return all([rule.budget > 0 for rule in rules])


def do_migration(group_i: GROUP, group_j: GROUP, no_migrant_tuples: int):
    '''Migrate tuples from group i to group j. Migrant tuples' indices figured out in migrant indices'''
    # kept_tuples_indices = list(set(range(len(group_i.origin_tuples))) - set(migrant_tuples_indices)) 
    # Move tuples to another group
    for tuple_to_move in group_i.origin_tuples[:no_migrant_tuples]:
        convert_quasi_attributes(tuple_to_move, group_j)
        group_j.received_tuples.append(tuple_to_move)

    group_i.origin_tuples = group_i.origin_tuples[no_migrant_tuples:]


def m3ar_modified_algo(D, R_initial, output_file_name='m3ar_modified.data', k=DESIRED_K):    
    start_time = time.time()
    # First construct R care
    R_care = construct_r_care(R_initial)    
    print('R CARE LENGTH =', len(R_care))
    # pprint_rule_set(R_care)
    rule_budgets = [rule.budget for rule in R_care]
    print('R CARE BUDGETS AT INITIAL:')
    print(rule_budgets)

    print('===============================================================================')    
    # Build groups from the dataset then split G into 2 sets of groups: safe groups SG and unsafe groups UG
    # Sort groups in UG and SG by length ascendingly
    GROUPS, SG, UG, _, _ = build_groups(D, R_care=R_care, k=k)
    print('THERE ARE {} SAFE GROUPS AND {} UNSAFE GROUPS'.format(len(SG), len(UG)))
    # cal_number_of_free_tuples(GROUPS, k)
    free_tuples = generate_free_tuples(SG, k)
    SelG = None
    loop_iteration = 0
    # STAGE 1: Pick free tuples into unsafe groups
    while len(UG) > 0 and len(free_tuples) > 0 and all_rules_having_positive_budgets(R_care):
        loop_iteration += 1
        if SelG is None:
            # MODIFICATION 1: UG was already sorted by group length. Process the UG groups one by one, start with the unsafe group with the longest length
            # The longer the group length, the more chance it will soon become a safe group
            SelG = UG.pop(0)
            # print('LOOP ITERATION {}. Pop the unsafe group with the longest length. SelG index: {}. SelG length: {}'.format(loop_iteration, SelG.index, group_length(SelG)))
            # print('Rules budget now are:')
            # print([rule.budget for rule in R_care])

            no_tuples_needed_to_become_a_safe_group = k - group_length(SelG)
            no_tuples_picked = 0
            free_tuples_to_remove = []
            for data_tuple in free_tuples:
                source_group = find_group(data_tuple.group_index, GROUPS)
                R_affected = construct_r_affected_by_a_migration(source_group, SelG)
                if all_rules_having_positive_budgets(R_affected):
                    for rule in R_affected:
                        rule.budget -= 1
                    # Perform a migration of this free tuple to the destination group                           
                    convert_quasi_attributes(data_tuple, SelG)
                    SelG.received_tuples.append(data_tuple)
                    # print('Data tuple picked is from group {}'.format(data_tuple.group_index))
                    source_group.origin_tuples.remove(data_tuple)
                    free_tuples_to_remove.append(data_tuple)
                    no_tuples_picked += 1
                    if no_tuples_picked == no_tuples_needed_to_become_a_safe_group:
                        # Pick enough, break to process the next unsafe group                        
                        add_group(SelG, SG)
                        remove_group(SelG, UG)
                        # print('Number of safe groups now is: {}. Number of free tuples is: {}'.format(len(SG), len(free_tuples)))
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
                    R_affected = construct_r_affected_by_a_migration(source_group, SelG)
                    free_tuples.append(data_tuple)
                    add_group(SelG, UG)
                    remove_group(SelG, SG)
                    for rule in R_affected:
                        rule.budget += 1

                SelG = None

    print('TOTAL LOOPS: {}'.format(loop_iteration))
    print('AFTER STAGE 1')
    print('NUMBER OF SAFE GROUPS AND UNSAFE GROUPS: {}, {}'.format(len(SG), len(UG)))
    print('TOTAL NUMBER OF TUPLES IN SAFE GROUPS: {}'.format(sum(group_length(group) for group in GROUPS if group_length(group) >= k)))
    print('TOTAL NUMBER OF TUPLES IN UNSAFE GROUPS: {}'.format(sum(group_length(group) for group in GROUPS if group_length(group) < k)))
    print('NUMBER OF FREE TUPLES: {}'.format(len(free_tuples)))
    rule_budgets = [rule.budget for rule in R_care]
    print('R CARE BUDGETS AFTER STAGE 1:')
    print(rule_budgets)

    # STAGE 2: PROCESS ONE BY ONE IN THE UNSAFE GROUP, START WITH GROUP WITH SHORT LENGTH
    # UG = sorted(UG, key=lambda gr: group_length(gr))
    # UG_SMALL_DISPERSED, UG_BIG_DISPERSED = [], []
    # for unsafe_group in UG:
    #     if group_length(unsafe_group) <= k / 2:  # With small group disperse them
    #         st = time.time()
    #         dst_group, R_affected = find_group_to_move_dispersing(unsafe_group, SG)
    #         print('Find group to move dispersing step takes', time.time() - st, 'seconds', 'Number of safe groups:', len(SG))
    #         if dst_group:
    #             for rule in R_affected:
    #                 rule.budget -= 1                
    #             print('DISPERSE SMALL GROUP', unsafe_group.index, '- LENGTH BEFORE:', group_length(unsafe_group))
    #             for data_tuple in unsafe_group.origin_tuples:
    #                 convert_quasi_attributes(data_tuple, dst_group)
    #                 dst_group.received_tuples.append(data_tuple)                                    
    #             for data_tuple in unsafe_group.received_tuples:
    #                 convert_quasi_attributes(data_tuple, dst_group)
    #                 dst_group.received_tuples.append(data_tuple)
    #             unsafe_group.origin_tuples = []
    #             unsafe_group.received_tuples = []
    #             print('DISPERSE SMALL GROUP', unsafe_group.index, '- LENGTH AFTER:', group_length(unsafe_group))
    #             # remove_group(unsafe_group, UG)
    #             UG_SMALL_DISPERSED.append(unsafe_group)
    #         else:
    #             print('Cannot find any group for group {} to disperse to'.format(unsafe_group.index))
    #     else:
    #         no_tuples_needed_to_become_a_safe_group = k - group_length(unsafe_group)
    #         picked_tuples = free_tuples[:no_tuples_needed_to_become_a_safe_group]
    #         if len(picked_tuples) < no_tuples_needed_to_become_a_safe_group:    # This means the free tuples set is now empty
    #             continue
    #         for data_tuple in picked_tuples:
    #             source_group = find_group(data_tuple.group_index, GROUPS)
    #             # Migrate these free tuples to this unsafe group to make it safe
    #             R_affected = construct_r_affected_by_a_migration(source_group, unsafe_group)
    #             for rule in R_affected:
    #                 rule.budget -= 1
    #             free_tuples.remove(data_tuple)
    #             source_group.origin_tuples.remove(data_tuple)
    #             convert_quasi_attributes(data_tuple, unsafe_group)
    #         print('UNSAFE BIG GROUP {} RECEIVED {} FREE TUPLES TO BECOME SAFE'.format(unsafe_group.index, no_tuples_needed_to_become_a_safe_group))
    #         unsafe_group.received_tuples.extend(picked_tuples)
    #         if is_safe_group(unsafe_group, k):  # Now becomes safe
    #             add_group(unsafe_group, SG)
    #             # remove_group(unsafe_group, UG)
    #             UG_BIG_DISPERSED.append(unsafe_group)

    # for g_sm in UG_SMALL_DISPERSED:
    #     print('Small group dispersed', g_sm.index, group_length(g_sm))
    #     remove_group(g_sm, UG)

    # for g_big in UG_BIG_DISPERSED:
    #     print('Big group dispersed', g_big.index, group_length(g_big))
    #     remove_group(g_big, UG)

    # print('AFTER STAGE 2')
    # print('NUMBER OF UNSAFE GROUPS AND SAFE GROUPS: {}, {}'.format(len([ug for ug in UG if group_length(ug) > 0]), len(SG)))
    # print('NUMBER OF UNSAFE GROUPS WITH LENGTH <= K/2 DISPERSED (MOVED TO A SAFE GROUP): {}'.format(len(UG_SMALL_DISPERSED)))
    # print('NUMBER OF UNSAFE GROUPS WITH LENGTH > K/2 DISPERSED (RECEIVED FREE TUPLES): {}'.format(len(UG_BIG_DISPERSED)))
    # print('TOTAL NUMBER OF SAFE GROUPS: {}'.format(len([group for group in GROUPS if group_length(group) >= k])))
    # print('TOTAL NUMBER OF UNSAFE GROUPS: {}'.format(len([group for group in GROUPS if 0 < group_length(group) < k])))
    # print('TOTAL NUMBER OF TUPLES IN SAFE GROUPS: {}'.format(sum(group_length(group) for group in GROUPS if group_length(group) >= k)))
    # print('TOTAL NUMBER OF TUPLES IN UNSAFE GROUPS: {}'.format(sum(group_length(group) for group in GROUPS if group_length(group) < k)))
    # print('NUMBER OF FREE TUPLES: {}'.format(len(free_tuples)))
    # # print('UG_SMALL_DISPERSED', [g.index for g in UG_SMALL_DISPERSED])
    # # print('UG_BIG_DISPERSED', [g.index for g in UG_BIG_DISPERSED])
    # rule_budgets = [rule.budget for rule in R_care]
    # print('R CARE BUDGETS:')
    # print(rule_budgets)

    # # STAGE 3: PROCESS ONE BY ONE IN THE REMAINING UNSAFE GROUP, START WITH GROUP WITH SHORT LENGTH
    # for unsafe_group in UG:
    #     # first_tuple_of_this_unsafe_group = unsafe_group.origin_tuples[0]
    #     dst_group, R_affected = find_group_to_move_dispersing(unsafe_group, SG)
    #     if dst_group:
    #         for rule in R_affected:
    #             rule.budget -= 1
    #         for data_tuple in unsafe_group.origin_tuples:
    #             convert_quasi_attributes(data_tuple, dst_group)
    #             dst_group.received_tuples.append(data_tuple)                                    
    #         for data_tuple in unsafe_group.received_tuples:
    #             convert_quasi_attributes(data_tuple, dst_group)
    #             dst_group.received_tuples.append(data_tuple)
    #         unsafe_group.origin_tuples = []
    #         unsafe_group.received_tuples = []
    #     else:
    #         print('Cannot find any group for group {} to disperse to'.format(unsafe_group.index))

    # print('TOTAL NUMBER OF TUPLES IN SAFE GROUPS: {}'.format(sum(group_length(group) for group in GROUPS if group_length(group) >= k)))
    # # print('TOTAL NUMBER OF TUPLES IN SAFE GROUPS 2: {}'.format(sum(group_length(group) for group in SG)))
    # print('TOTAL NUMBER OF TUPLES IN UNSAFE GROUPS: {}'.format(sum(group_length(group) for group in GROUPS if group_length(group) < k)))
    # # print('TOTAL NUMBER OF TUPLES IN UNSAFE GROUPS 2: {}'.format(sum(group_length(group) for group in UG)))

    UG_SMALL, UG_BIG = [], []
    for g in UG:
        if group_length(g) <= k/2:
            UG_SMALL.append(g)
        else:
            UG_BIG.append(g)
    # SIMILAR TO M3AR
    UM = []  # Set of groups that cannot migrate member with other groups
    SelG = None
    loop_iteration = 0
    UG.sort(key=lambda group: group_length(group))
    while (len(UG) > 0) or (SelG):
        loop_iteration += 1
        print('START LOOP ITERATION {}. UG length: {}. SG length: {}. UM length: {}'.format(loop_iteration, len(UG), len(SG), len(UM)))
        if SelG is None:  # Randomly pick a group SelG from unsafe groups set
            # print('LOOP ITERATION {}. SelG is None, randomly pick one'.format(loop_iteration))
            SelG = UG.pop(0)
            # remove_group(SelG, UG)
            if group_length(SelG) <= k/2:
                remove_group(SelG, UG_SMALL)
            else:
                remove_group(SelG, UG_BIG)
        
        # print('LOOP ITERATION {}. SelG: {} ({})'.format(loop_iteration, SelG.index, group_length(SelG)))

        # Find the most appropriate group g in UG and SG to perform migration with SelG
        if group_length(SelG) <= k/2:
            remaining_groups = UG_BIG + UG_SMALL + SG
        else:
            remaining_groups = UG_SMALL + UG_BIG + SG
        result_find_migration = find_group_to_migrate(R_care, SelG, remaining_groups, k)        
        # If cannot find such a group, add it to the unmigrant UM set
        if result_find_migration is None:
            # print('LOOP ITERATION {}: NO RESULT FOR MIGRATION'.format(loop_iteration))
            add_group(SelG, UM)
            SelG = None
        else:   # If we can find a migration operation
            # print('LOOP ITERATION {}: PREPARE MIGRATION GROUP {} ({}) => GROUP {} ({}): {} TUPLES'.format(loop_iteration, result_find_migration.group_i.index, group_length(result_find_migration.group_i), result_find_migration.group_j.index, group_length(result_find_migration.group_j), result_find_migration.no_migrant_tuples))
            # g is the other group in the migration operation
            g = result_find_migration.group_i if result_find_migration.group_i.index != SelG.index else result_find_migration.group_j
            # Perform a migration operation
            do_migration(result_find_migration.group_i, result_find_migration.group_j, result_find_migration.no_migrant_tuples)
            # print('LOOP ITERATION {}: AFTER MIGRATION: SelG: {} ({}). g: {} ({})'.format(loop_iteration, SelG.index, group_length(SelG), g.index, group_length(g)))
            for rule in result_find_migration.R_affected:
                rule.budget -= 1

            if group_length(SelG) == 0:
                remove_group(SelG, UG)
                remove_group(SelG, UG_SMALL)
                remove_group(SelG, UG_BIG)

            if group_length(g) == 0:
                remove_group(g, UG)
                remove_group(g, UG_SMALL)
                remove_group(g, UG_BIG)

            if group_length(SelG) == 0 and group_length(g) == 0:
                SelG = None
                # print('LOOP ITERATION {}: BOTH GROUPS SelG AND g NOW HAVE 0 TUPLES'.format(loop_iteration))
                # print('LOOP ITERATION {} FINISHES IN {} SECONDS'.format(loop_iteration, time.time() - st))
                continue

            # Check if now we have a safe group in the pair (SelG, g) or not. 
            # If there is one, collect it and add it to the safe group
            if is_safe_group(SelG, k):
                add_group(SelG, SG)
                remove_group(SelG, UG)
                remove_group(SelG, UG_SMALL)
                remove_group(SelG, UG_BIG)
                # print('LOOP ITERATION {}: MOVE GROUP {} TO SG'.format(loop_iteration, SelG.index))

            if is_safe_group(g, k):
                add_group(g, SG)
                # print('LOOP ITERATION {}: MOVE GROUP {} TO SG'.format(loop_iteration, g.index))
                remove_group(g, UG)
                remove_group(g, UG_SMALL)
                remove_group(g, UG_BIG)

            # print('LOOP ITERATION {}: CHECK SAFE - SelG: {} ({},{}) - g: {} ({},{})'.format(loop_iteration, SelG.index, is_safe_group(SelG, k), 'Empty' if group_length(SelG) == 0 else '', g.index, is_safe_group(g, k), 'Empty' if group_length(g) == 0 else ''))

            # Handle which group next? If there is any unsafe group in the pair, continue with it
            if is_unsafe_group(SelG, k):
                pass    # Keep handling SelG
            elif is_unsafe_group(g, k):    # Continue with g
                SelG = g
                remove_group(g, UG)
            else:
                # The next iteration we will choose another group to process
                SelG = None

            # print('LOOP ITERATION {} FINISHES IN {} SECONDS'.format(loop_iteration, time.time() - st))

    print('TOTAL LOOPS: {}. UG length: {}. SG length: {}. UM length: {}\n'.format(loop_iteration, len(UG), len(SG), len(UM)))    
    print('TOTAL NUMBER OF TUPLES IN SAFE GROUPS: {}'.format(sum(group_length(group) for group in GROUPS if group_length(group) >= k)))
    print('TOTAL NUMBER OF TUPLES IN SAFE GROUPS 2: {}'.format(sum(group_length(group) for group in SG)))
    print('TOTAL NUMBER OF TUPLES IN UNSAFE GROUPS: {}'.format(sum(group_length(group) for group in GROUPS if group_length(group) < k and group not in UM)))
    print('TOTAL NUMBER OF TUPLES IN UNSAFE GROUPS 2: {}'.format(sum(group_length(group) for group in UG)))
    print('TOTAL NUMBER OF TUPLES IN UM: {}'.format(sum(group_length(group) for group in UM)))
    print('=======================================')
    print('=======================================')
    print('=======================================')
    print('START TO DISPERSE', len(UM), 'UM GROUPS')
    print('NUMBER OF UM GROUPS WITH LENGTH > 0:', sum(1 for group in UM if group_length(group) > 0))
    while len(UM) > 0: # Disperse
        g_um = UM.pop(0)
        if group_length(g_um) > 0:  # Just consider group with length > 0
            disperse(R_care, g_um, GROUPS, SG, UM)

    # print('AFTER DISPERSING: NUMBER OF UM GROUPS WITH LENGTH > 0:', sum(1 for group in UM if group_length(group) > 0))
    # print('FINAL RESULTS: UG length: {}. SG length: {}. UM length: {}\n'.format(len(UG), len(SG), len(UM)))    
    # print('TOTAL NUMBER OF TUPLES IN SAFE GROUPS: {}'.format(sum(group_length(group) for group in GROUPS if group_length(group) >= k)))
    # print('TOTAL NUMBER OF TUPLES IN SAFE GROUPS 2: {}'.format(sum(group_length(group) for group in SG)))
    # print('TOTAL NUMBER OF TUPLES IN UNSAFE GROUPS: {}'.format(sum(group_length(group) for group in GROUPS if group_length(group) < k)))
    # print('TOTAL NUMBER OF TUPLES IN UNSAFE GROUPS 2: {}'.format(sum(group_length(group) for group in UG)))
    # print('TOTAL NUMBER OF TUPLES IN UM: {}'.format(sum(group_length(group) for group in UM)))    
    total_time = time.time() - start_time
    eval_results(R_initial, GROUPS, output_file_name, total_time, k=k)


if __name__ == '__main__':
    if len(sys.argv) > 3:
        data_file_path, initial_rules_path, k = sys.argv[1], sys.argv[2], int(sys.argv[3])
        log_to_file = True
    else:
        data_file_path = 'dataset/adult-prep.data'
        initial_rules_path = 'adult-prep-rules-picked.data'
        k = 10
        log_to_file = False

    if log_to_file:
        sys.stdout = open("log/modified_algo_results_k_" + str(k) + ".log", "w")
    # A dataset reaches k-anonymity if total risks of all groups equals to 0
    # A Member Migration operation g(i)-T-g(j) is valuable when the risk of data is decreased after performing that Member Migration operation.
    D = pandas.read_csv(data_file_path, names=RETAINED_DATA_COLUMNS, index_col=False, skipinitialspace=True)
    dataset_length = D.shape[0]
    print('DATASET LENGTH=', dataset_length)
    print('MIN_SUP=', MIN_SUP)
    print('MIN_CONF=', MIN_SUP)
    print('K=', k)
    MIN_SUP = MIN_SUP * dataset_length
    R_initial = []
    with open(initial_rules_path, 'rb') as f:
        R_initial = pickle.load(f)
    # print('R initial', R_initial)
    output_file_name = 'out_mod_m3ar_' + str(k) + '_' + data_file_path.split('/')[-1].split('.')[0] + '.data'
    m3ar_modified_algo(D, R_initial, output_file_name, k)

    sys.stdout.close()
