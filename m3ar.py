import csv
import pandas
import math
import random
import time
import traceback
import sys
import pickle
from dataclasses import dataclass
from itertools import combinations
from common import *
from eval import eval_results


R_AFFECTED = {}


def group_risk(a_group, k=DESIRED_K):
    group_len = group_length(a_group)
    if group_len >= k:
        return 0
    else:
        return 2*k - group_len


def calc_risk_reduction(group_i: GROUP, group_j: GROUP, no_migrant_tuples: int, k=DESIRED_K):
    '''Calculate the risk reduction in case performing 
    a migration operation of <no_migrant_tuples> from group i to group j'''
    risk_before = group_risk(group_i, k) + group_risk(group_j, k)    
    group_i_length_after = group_length(group_i) - no_migrant_tuples
    group_i_risk_after = 0 if group_i_length_after >= k or group_i_length_after == 0 else 2*k - group_i_length_after
    group_j_length_after = group_length(group_j) + no_migrant_tuples
    group_j_risk_after = 0 if group_j_length_after >= k  or group_j_length_after == 0 else 2*k - group_j_length_after
    risk_reduction = risk_before - (group_i_risk_after + group_j_risk_after)
    most_useful = False
    if is_unsafe_group(group_i, k) and is_unsafe_group(group_j, k) and risk_reduction == risk_before:
        most_useful = True
    return risk_reduction, most_useful


# POLICIES
'''
POLICY 1. A k-unsafe group once has received tuple(s), it can only
continue receiving tuple(s); otherwise, as its tuple(s) migrate
to another group, it can only continue giving its tuple(s) to
other groups. The policy does not apply to k-safe groups. ONLY for k-unsafe groups
'''


def has_group_received_tuples(a_group: GROUP):
    return len(a_group.received_tuples) > 0


def has_group_given_tuples(a_group: GROUP):
    return len(a_group.origin_tuples) < a_group.origin_len


def choose_group_tuples_for_migration(R_care: list, group_i: GROUP, no_tuples: int, group_j: GROUP):  
    '''
    Return indices of tuples selected for a migration  operation of <no_tuples> tuples from <a_group>
    '''
    comb = combinations(range(len(group_i.origin_tuples)), no_tuples)

    for a_comb in list(comb):   # Encounter a combination that is good get it
        tuples_consider_to_move = [group_i.origin_tuples[i] for i in a_comb]
        R_affected = construct_r_affected_by_a_migration(R_care, tuples_consider_to_move, group_j)
        budgets_all_rules_affected_positive = all(rule.budget > 0 for rule in R_affected)
        if budgets_all_rules_affected_positive:
            return a_comb, R_affected

    return None, []
    

'''
POLICY 2. g(i)⎯T->g(j) and ∀t∈T → (∀r ∈ R(t),g(i)->g(j)->Budget(r) > 0)
If migrating tuples T from group i to group j, all rules affected must have positive budgets
'''


def migration_check_budget_all_rules_affected_positive(R_care: list, T: list, group_j: GROUP):
    '''Check if budget of the rules affected by a migration operation positive'''
    R_affected = construct_r_affected_by_a_migration(R_care, T, group_j)
    return all(rule.budget > 0 for rule in R_affected), R_affected


def budgets_of_rules_affected_by_migrating_a_tuple(R_care, t: DATA_TUPLE, group_j: GROUP):
    '''Calculate budgets of rules affected by migrating a data tuple to another group'''
    R_affected = construct_r_affected_by_a_migration(R_care, [t], group_j)
    return [rule.budget for rule in R_affected]


def build_rules_affected_by_perform_migration(R_care, GROUPS):
    result = {}
    for group_outer in GROUPS:
        for group_inner in GROUPS:
            if group_outer.index != group_inner.index:
                key = '{}_{}'.format(group_outer.index, group_inner.index)
                print(key)
                result[key] = construct_r_affected_by_a_migration(R_care, [group_outer.origin_tuples[0]], group_inner)        

    return result


'''
POLICY 3. Calculate number of migrant tuples in a migration operation
'''


def cal_number_of_migrant_tuples(group_i, group_j, k=DESIRED_K):
    '''Calculate number of tuples that can be migrated from group i to group j'''
    if is_unsafe_group(group_i, k) and is_unsafe_group(group_j, k): # Group i and group j are both unsafe
        return min(group_length(group_i), k - group_length(group_j))
    elif is_safe_group(group_i, k): # Group i safe, group j unsafe
        return min(k - group_length(group_j), group_length(group_i) - k, len(group_i.origin_tuples))
    else:   # Group i is unsafe
        return group_length(group_i)

            
# END OF POLICIES
def apply_policies(R_care, group_i, group_j, k=DESIRED_K):
    '''
    Consider policies to perform a member migration from group i to group j
    '''
    ableToMigrate, no_migrant_tuples, risk_reduction, time_elapsed, R_affected, most_useful = False, -1, -9999, -1, [], False
    # start = time.time()
    # POLICY 1
    if is_unsafe_group(group_i, k) and has_group_received_tuples(group_i):
        # print('GROUP {} HAS RECEIVED TUPLES BEFORE, CANNOT GIVE TO GROUP {}'.format(group_i.index, group_j.index))
        return ableToMigrate, group_i, group_j, no_migrant_tuples, risk_reduction, time_elapsed, R_affected, most_useful
    # Group j cannot receive more tuples because it is an unsafe group and it has given its tuples to another before
    if is_unsafe_group(group_j, k) and has_group_given_tuples(group_j):
        # print('GROUP {} HAS GIVEN TUPLES BEFORE, CANNOT RECEIVE FROM GROUP {}'.format(group_j.index, group_i.index))
        return ableToMigrate, group_i, group_j, no_migrant_tuples, risk_reduction, time_elapsed, R_affected, most_useful
    # POLICY 3
    no_migrant_tuples = cal_number_of_migrant_tuples(group_i, group_j, k)
    # POLICY 2
    if no_migrant_tuples > 0:
        # Check budget of rules affected
        tuples_consider_to_move = group_i.origin_tuples[:no_migrant_tuples]
        R_affected = construct_r_affected_by_a_migration(R_care, [tuples_consider_to_move[0]], group_j)
        # R_affected = R_AFFECTED['{}_{}'.format(group_i.index, group_j.index)]
        budgets_all_rules_affected_positive = all(rule.budget > 0 for rule in R_affected)
        if budgets_all_rules_affected_positive:
            ableToMigrate = True
            risk_reduction, most_useful = calc_risk_reduction(group_i, group_j, no_migrant_tuples, k)

    # time_elapsed = time.time() - start
    return ableToMigrate, group_i, group_j, no_migrant_tuples, risk_reduction, time_elapsed, R_affected, most_useful


def do_migration(group_i: GROUP, group_j: GROUP, no_migrant_tuples: int):
    '''Migrate tuples from group i to group j. Migrant tuples' indices figured out in migrant indices'''
    # kept_tuples_indices = list(set(range(len(group_i.origin_tuples))) - set(migrant_tuples_indices)) 
    # Move tuples to another group
    for tuple_to_move in group_i.origin_tuples[:no_migrant_tuples]:
        convert_quasi_attributes(tuple_to_move, group_j)
        group_j.received_tuples.append(tuple_to_move)

    group_i.origin_tuples = group_i.origin_tuples[no_migrant_tuples:]


# Apply policies to perform a useful migration operation
def find_group_to_migrate(R_care: list, selected_group: GROUP, remaining_groups: list, k: DESIRED_K):
    results = []
    print('FIND A GROUP TO PERFORM MIGRATION IN {} REMAINING GROUPS'.format(len(remaining_groups)))
    st = time.time()
    for group in remaining_groups:        
        if group.index != selected_group.index:
            # Start to apply policies
            factors = apply_policies(R_care, selected_group, group, k)
            # Check if the last migration is most useful
            if factors[-1]:
                print('FIND THE MOST USEFUL: GROUP {}. BREAK!'.format(factors[2].index))
                resultsDataFrame = pandas.DataFrame([factors[:-1]], columns=['ableToMigrate', 'group_i', 'group_j', 'no_migrant_tuples', 'risk_reduction', 'time_elapsed', 'R_affected'])
                print('RUN TIME TO FIND A GROUP TO PERFORM MIGRATION: {} seconds'.format(time.time() - st))
                return resultsDataFrame.iloc[0]
            factors_reverse = apply_policies(R_care, group, selected_group, k)
            if factors_reverse[-1]:
                print('FIND THE MOST USEFUL: GROUP {}. BREAK!'.format(factors[2].index))
                resultsDataFrame = pandas.DataFrame([factors_reverse[:-1]], columns=['ableToMigrate', 'group_i', 'group_j', 'no_migrant_tuples', 'risk_reduction', 'time_elapsed', 'R_affected'])
                print('RUN TIME TO FIND A GROUP TO PERFORM MIGRATION: {} seconds'.format(time.time() - st))
                return resultsDataFrame.iloc[0]
            # First element of factors and factors_reverse: able_to_migrate True/False
            if factors[0]:
                results.append(factors)
            if factors_reverse[0]:
                results.append(factors_reverse)    
    if len(results) == 0:   # There is no choice to move
        return None
    
    # Construct pandas DataFrame results
    resultsDataFrame = pandas.DataFrame(results, columns=['ableToMigrate', 'group_i', 'group_j', 'no_migrant_tuples', 'risk_reduction', 'time_elapsed', 'R_affected', 'most_useful'])    
    # Get the best migration selection
    resultsDataFrame.sort_values(by=['risk_reduction', 'no_migrant_tuples'], ascending=[False, True], inplace=True)
    # print('COMPARISON RESULTS', resultsDataFrame)
    print('RUN TIME TO FIND A GROUP TO PERFORM MIGRATION: {} seconds'.format(time.time() - st))
    return resultsDataFrame.iloc[0]


# Migrate the data tuple into the most useful group of a safe group
def find_group_to_move_dispersing(R_care: list, t: DATA_TUPLE, SG: list):
    min_no_rule_budgets = 99999
    result = None
    for considering_dst_group in SG:
        # Start to apply policies
        rule_budgets = budgets_of_rules_affected_by_migrating_a_tuple(R_care, t, considering_dst_group)
        no_rules_with_negative_budget = sum(1 for budget in rule_budgets if budget < 0)

        if no_rules_with_negative_budget < min_no_rule_budgets:
            result = considering_dst_group
            min_no_rule_budgets = no_rules_with_negative_budget            

    return result


def do_disperse_migration(R_care: list, src_group: GROUP, data_tuple: DATA_TUPLE, dst_group: GROUP, is_return=False):    
    convert_quasi_attributes(data_tuple, dst_group)
    R_affected = construct_r_affected_by_a_migration(R_care, [data_tuple], dst_group)
    if not is_return:
        src_group.origin_tuples.remove(data_tuple)
        dst_group.received_tuples.append(data_tuple)
        for rule in R_affected:
            rule.budget -= 1
    else:
        src_group.received_tuples.remove(data_tuple)
        dst_group.origin_tuples.append(data_tuple)
        for rule in R_affected:
            rule.budget += 1


def disperse(R_care: list, um_group: GROUP, GROUPS: list, SG: list, UM: list):
    '''
    This group is still unsafe and it contains tuples migrated from other groups
    In loop we cannot find another group to perform migration with this group
    Disperse, return or move (Giai tan) tuples that have been migrated into this group to other ones
    '''
    print('DISPERSE GROUP', um_group.index)
    # pprint_groups([um_group])
    # For all receiving tuples, return them to the source group
    while len(um_group.received_tuples) > 0:
        data_tuple = um_group.received_tuples[0]
        source_group_of_this_tuple = find_group(data_tuple.group_index, GROUPS)
        do_disperse_migration(R_care, um_group, data_tuple, source_group_of_this_tuple, is_return=True)
        print('DISPERSE: GROUP {} RETAKES THE DATA TUPLE {}'.format(source_group_of_this_tuple.index, data_tuple.index))        
        # If the source group now has only 1 tuple, add it to UM to be processed
        if group_length(source_group_of_this_tuple) == 1:
            add_group(source_group_of_this_tuple, UM)

    while len(um_group.origin_tuples) > 0:
        data_tuple = um_group.origin_tuples[0]
        # Find the most appropriate group g in SG to perform migration
        dst_group_to_migrate = find_group_to_move_dispersing(R_care, data_tuple, SG)
        if dst_group_to_migrate:
            print('DISPERSE: GROUP {} TAKES THE DATA TUPLE {} FROM GROUP {}'.format(dst_group_to_migrate.index, data_tuple.index, um_group.index))
            do_disperse_migration(R_care, um_group, data_tuple, dst_group_to_migrate, is_return=False)                
        else:
            print('DISPERSE: CANNOT FIND ANY GROUP TO MOVE TUPLES FROM GROUP {}'.format(um_group.index))


def m3ar_algo(D, R_initial, output_file_name, k=DESIRED_K):
    start_time = time.time()
    # Build groups from the dataset then split G into 2 sets of groups: safe groups SG and unsafe groups UG
    GROUPS, SG, UG, UG_SMALL, UG_BIG = build_groups(D, k=k)
    print('TOTAL NUMBER OF TUPLES IN SAFE GROUPS: {}'.format(sum(group_length(group) for group in SG)))
    print('TOTAL NUMBER OF TUPLES IN UNSAFE GROUPS: {}'.format(sum(group_length(group) for group in UG)))
    UM = []  # Set of groups that cannot migrate member with other groups
    print('K =', k)
    print('NUMBER OF UNSAFE GROUPS AND SAFE GROUPS:', len(UG), len(SG))
    R_care = construct_r_care(R_initial)  # List of cared rules
    R_care = R_care[:5]
    for r in R_care:
        r.budget = rule_budget(r)
    print('LENGTH OF R_care', len(R_care))
    print('===============================================================================')
    # Build R_affected
    # global R_AFFECTED
    # R_AFFECTED = build_rules_affected_by_perform_migration(R_care, GROUPS)
    # print('LENGTH R_AFFECTED', len(R_AFFECTED))
    SelG = None
    loop_iteration = 0
    while (len(UG) > 0) or (SelG):
        st = time.time()
        loop_iteration += 1
        print('START LOOP ITERATION {}. UG length: {}. SG length: {}. UM length: {}'.format(loop_iteration, len(UG), len(SG), len(UM)))
        if SelG is None:  # Randomly pick a group SelG from unsafe groups set
            print('LOOP ITERATION {}. SelG is None, randomly pick one'.format(loop_iteration))   
            SelG = random.choice(UG)
            remove_group(SelG, UG)
        
        print('LOOP ITERATION {}. SelG: {} ({})'.format(loop_iteration, SelG.index, group_length(SelG)))

        # Find the most appropriate group g in UG and SG to perform migration with SelG
        if group_length(SelG) <= k/2:
            remaining_groups = UG_BIG + UG_SMALL + SG
        else:
            remaining_groups = UG_SMALL + UG_BIG + SG
        result_find_migration = find_group_to_migrate(R_care, SelG, remaining_groups, k)        
        # If cannot find such a group, add it to the unmigrant UM set
        if result_find_migration is None:
            print('LOOP ITERATION {}: NO RESULT FOR MIGRATION'.format(loop_iteration))
            add_group(SelG, UM)
            SelG = None
        else:   # If we can find a migration operation
            print('LOOP ITERATION {}: PREPARE MIGRATION GROUP {} ({}) => GROUP {} ({}): {} TUPLES'.format(loop_iteration, result_find_migration.group_i.index, group_length(result_find_migration.group_i), result_find_migration.group_j.index, group_length(result_find_migration.group_j), result_find_migration.no_migrant_tuples))
            # g is the other group in the migration operation
            g = result_find_migration.group_i if result_find_migration.group_i.index != SelG.index else result_find_migration.group_j
            # Perform a migration operation
            do_migration(result_find_migration.group_i, result_find_migration.group_j, result_find_migration.no_migrant_tuples)
            print('LOOP ITERATION {}. AFTER MIGRATION: SelG: {} ({}). g: {} ({})'.format(loop_iteration, SelG.index, group_length(SelG), g.index, group_length(g)))
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
                print('LOOP ITERATION {}: BOTH GROUPS SelG AND g NOW HAVE 0 TUPLES'.format(loop_iteration))
                print('LOOP ITERATION {} FINISHES IN {} SECONDS'.format(loop_iteration, time.time() - st))
                continue

            # Check if now we have a safe group in the pair (SelG, g) or not. 
            # If there is one, collect it and add it to the safe group
            if is_safe_group(SelG, k):
                add_group(SelG, SG)
                remove_group(SelG, UG)
                remove_group(SelG, UG_SMALL)
                remove_group(SelG, UG_BIG)
                print('LOOP ITERATION {}: MOVE GROUP {} TO SG'.format(loop_iteration, SelG.index))

            if is_safe_group(g, k):
                add_group(g, SG)
                print('LOOP ITERATION {}: MOVE GROUP {} TO SG'.format(loop_iteration, g.index))
                remove_group(g, UG)
                remove_group(g, UG_SMALL)
                remove_group(g, UG_BIG)

            print('LOOP ITERATION {}: CHECK SAFE - SelG: {} ({},{}) - g: {} ({},{})'.format(loop_iteration, SelG.index, is_safe_group(SelG, k), 'Empty' if group_length(SelG) == 0 else '', g.index, is_safe_group(g, k), 'Empty' if group_length(g) == 0 else ''))

            # Handle which group next? If there is any unsafe group in the pair, continue with it
            if is_unsafe_group(SelG, k):
                pass    # Keep handling SelG
            elif is_unsafe_group(g, k):    # Continue with g
                SelG = g
                remove_group(g, UG)
            else:
                # The next iteration we will choose another group to process
                SelG = None

            print('LOOP ITERATION {} FINISHES IN {} SECONDS'.format(loop_iteration, time.time() - st))

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

    print('AFTER DISPERSING: NUMBER OF UM GROUPS WITH LENGTH > 0:', sum(1 for group in UM if group_length(group) > 0))
    print('FINAL RESULTS: UG length: {}. SG length: {}. UM length: {}\n'.format(len(UG), len(SG), len(UM)))    
    print('TOTAL NUMBER OF TUPLES IN SAFE GROUPS: {}'.format(sum(group_length(group) for group in GROUPS if group_length(group) >= k)))
    print('TOTAL NUMBER OF TUPLES IN SAFE GROUPS 2: {}'.format(sum(group_length(group) for group in SG)))
    print('TOTAL NUMBER OF TUPLES IN UNSAFE GROUPS: {}'.format(sum(group_length(group) for group in GROUPS if group_length(group) < k)))
    print('TOTAL NUMBER OF TUPLES IN UNSAFE GROUPS 2: {}'.format(sum(group_length(group) for group in UG)))
    print('TOTAL NUMBER OF TUPLES IN UM: {}'.format(sum(group_length(group) for group in UM)))    

    eval_results(R_initial, GROUPS, output_file_name, start_time)


if __name__ == '__main__':
    # sys.stdout = open("log/m3ar_results_k_" + k + "".log", "w")
    if len(sys.argv) > 3:
        data_file_path, initial_rules_path, k = sys.argv[1], sys.argv[2], int(sys.argv[3])
    else:
        data_file_path = 'dataset/adult-prep.data'
        initial_rules_path = 'adult-prep-rules-picked.data'
        k = 10
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
    output_file_name = 'out_m3ar_k_' + str(k) + '_' + data_file_path.split('/')[-1].split('.')[0] + '.data'
    m3ar_algo(D, R_initial, output_file_name, k)

    sys.stdout.close()
