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


def group_risk(a_group):
    group_len = group_length(a_group)
    if group_len >= DESIRED_K:
        return 0
    else:
        return 2*DESIRED_K - group_len


def calc_risk_reduction(group_i: GROUP, group_j: GROUP, no_migrant_tuples: int):
    '''Calculate the risk reduction in case performing 
    a migration operation of <no_migrant_tuples> from group i to group j'''
    risk_before = group_risk(group_i) + group_risk(group_j)
    group_i_length_after = group_length(group_i) - no_migrant_tuples
    group_i_risk_after = 0 if group_i_length_after >= DESIRED_K else 2*DESIRED_K - group_i_length_after
    group_j_length_after = group_length(group_j) + no_migrant_tuples
    group_j_risk_after = 0 if group_j_length_after >= DESIRED_K else 2*DESIRED_K - group_j_length_after
    return risk_before - (group_i_risk_after + group_j_risk_after)


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


def choose_group_tuples_for_migration(R_care: list, group_i: GROUP, no_tuples: int, group_j):  
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


'''
POLICY 3. Calculate number of migrant tuples in a migration operation
'''


# Migrate tuples from group i to group j
def cal_number_of_migrant_tuples(group_i, group_j, migration_direction='l2r'):
    if is_unsafe_group(group_j):
        if migration_direction == 'l2r':
            return min(group_length(group_i), DESIRED_K - group_length(group_j))
        else:
            return min(group_length(group_j), DESIRED_K - group_length(group_i))
    else:
        if migration_direction == 'l2r':
            return group_length(group_i)
        else:
            return min(DESIRED_K - group_length(group_i),
                                    group_length(group_j) - DESIRED_K, len(group_j.origin_tuples))
            
# END OF POLICIES
def apply_policies(R_care, group_i, group_j, migration_direction='l2r'):
    '''
    Consider policies to perform a member migration from group i to group j
    '''
    ableToMigrate, migrant_tuples_indices, no_migrant_tuples, risk_reduction, time_elapsed, R_affected = False, [], -1, -9999, -1, []
    start = time.time()
    # POLICY 1
    if is_unsafe_group(group_i) and has_group_received_tuples(group_i):
        # print('GROUP {} HAS RECEIVED TUPLES BEFORE, CANNOT GIVE TO GROUP {}'.format(group_i.index, group_j.index))
        return ableToMigrate, group_i, group_j, migrant_tuples_indices, no_migrant_tuples, risk_reduction, time_elapsed, R_affected
    # Group j cannot receive more tuples because it is an unsafe group and it has given its tuples to another before
    if is_unsafe_group(group_j) and has_group_given_tuples(group_j):
        # print('GROUP {} HAS GIVEN TUPLES BEFORE, CANNOT RECEIVE FROM GROUP {}'.format(group_j.index, group_i.index))
        return ableToMigrate, group_i, group_j, migrant_tuples_indices, no_migrant_tuples, risk_reduction, time_elapsed, R_affected
    # POLICY 3
    no_migrant_tuples = cal_number_of_migrant_tuples(group_i, group_j, migration_direction)
    # POLICY 2
    if no_migrant_tuples > 0:
        # Select tuples from group i to satisfy the condition in which the budget all rules affected greater than 0
        migrant_tuples_indices, R_affected = choose_group_tuples_for_migration(R_care, group_i, no_migrant_tuples, group_j)
        if migrant_tuples_indices:  # if find a satisfied tuples
            ableToMigrate = True
            risk_reduction = calc_risk_reduction(group_i, group_j, no_migrant_tuples)

    time_elapsed = time.time() - start    
    return ableToMigrate, group_i, group_j, migrant_tuples_indices, no_migrant_tuples, risk_reduction, time_elapsed, R_affected


def do_migration(group_i: GROUP, group_j: GROUP, migrant_tuples_indices: list):
    '''Migrate tuples from group i to group j. Migrant tuples' indices figured out in migrant indices'''
    kept_tuples_indices = list(set(range(len(group_i.origin_tuples))) - set(migrant_tuples_indices)) 
    # Move tuples to another group
    for i in migrant_tuples_indices:
        tuple_to_move = group_i.origin_tuples[i]
        convert_quasi_attributes(tuple_to_move, group_j)
        group_j.received_tuples.append(tuple_to_move)

    # Group i now only has some tuples kept
    group_i.origin_tuples = [group_i.origin_tuples[kept_i] for kept_i in kept_tuples_indices]
    

# Apply policies to perform a useful migration operation
def find_group_to_migrate(R_care: list, selected_group: GROUP, UG: list, SG: list):
    remaining_groups = UG + SG
    results = []
    # i = 0
    print('FIND A GROUP TO PERFORM MIGRATION IN {} REMAINING GROUPS'.format(len(remaining_groups)))
    st = time.time()
    for group in remaining_groups:
        # if i % 200 == 0:
        #     print('DEBUG', i)
        if group.index != selected_group.index:
            # Start to apply policies
            factors = apply_policies(R_care, selected_group, group, 'l2r')
            factors_reverse = apply_policies(R_care, group, selected_group, 'r2l')
            if factors[0]:
                results.append(factors)
            if factors_reverse[0]:
                results.append(factors_reverse)
        # i += 1
    print('RUN TIME TO FIND A GROUP TO MIGRATION: {}'.format(time.time() - st))
    if len(results) == 0:   # There is no choice to move
        return None
    
    # Construct pandas DataFrame results
    resultsDataFrame = pandas.DataFrame(results, columns=['ableToMigrate', 'group_i', 'group_j', 'migrant_tuples_indices', 'no_migrant_tuples', 'risk_reduction', 'time_elapsed', 'R_affected'])    
    # Get the best migration selection
    resultsDataFrame.sort_values(by=['time_elapsed', 'risk_reduction', 'no_migrant_tuples'], ascending=[True, False, True])
    # print('COMPARISON RESULTS', resultsDataFrame)
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


def m3ar_algo(D, R_initial, output_file_name):
    start_time = time.time()
    # Build groups from the dataset then split G into 2 sets of groups: safe groups SG and unsafe groups UG
    GROUPS, SG, UG = build_groups(D)
    print('TOTAL NUMBER OF TUPLES IN SAFE GROUPS: {}'.format(sum(group_length(group) for group in SG)))
    print('TOTAL NUMBER OF TUPLES IN UNSAFE GROUPS: {}'.format(sum(group_length(group) for group in UG)))
    UM = []  # Set of groups that cannot migrate member with other groups
    print('K =', DESIRED_K)
    print('NUMBER OF UNSAFE GROUPS AND SAFE GROUPS:', len(UG), len(SG))
    R_care = construct_r_care(R_initial)  # List of cared rules
    for r in R_care:
        r.budget = rule_budget(r)
    print('LENGTH OF R_care', len(R_care))
    print('===============================================================================')
    SelG = None
    loop_iteration = 0
    while (len(UG) > 0) or (SelG):
        loop_iteration += 1
        print('LOOP ITERATION {}. UG length: {}. SG length: {}. UM length: {}. SelG index: {}'.format(loop_iteration, len(UG), len(SG), len(UM), SelG.index if SelG is not None else None))
        if SelG is None:  # Randomly pick a group SelG from unsafe groups set        
            SelG = random.choice(UG)
            remove_group(SelG, UG)
            print('LOOP ITERATION {}. Randomly pop SelG. SelG index: {}'.format(loop_iteration, SelG.index))

        # Find the most appropriate group g in UG and SG to perform migration with SelG
        result_find_migration = find_group_to_migrate(R_care, SelG, UG, SG)        
        # If cannot find such a group, add it to the unmigrant UM set
        if result_find_migration is None:
            print('LOOP ITERATION {}: NO RESULT FOR MIGRATION'.format(loop_iteration))
            add_group(SelG, UM)
            SelG = None
        else:   # If we can find a migration operation
            print('LOOP ITERATION {}: MIGRATION {} TUPLES FROM GROUP {} WITH LENGTH {} TO GROUP {} WITH LENGTH {}'.format(loop_iteration, result_find_migration.no_migrant_tuples, result_find_migration.group_i.index, group_length(result_find_migration.group_i), result_find_migration.group_j.index, group_length(result_find_migration.group_j)))
            # g is the other group in the migration operation
            g = result_find_migration.group_i if result_find_migration.group_i.index != SelG.index else result_find_migration.group_j
            # Perform a migration operation
            do_migration(result_find_migration.group_i, result_find_migration.group_j, result_find_migration.migrant_tuples_indices)
            for rule in result_find_migration.R_affected:
                rule.budget -= 1

            # Check if now we have a safe group in the pair (SelG, g) or not. 
            # If there is one, collect it and add it to the safe group
            if is_safe_group(SelG):
                add_group(SelG, SG)
                print('LOOP ITERATION {}: ADD GROUP {} TO SG'.format(loop_iteration, SelG.index))

            if is_safe_group(g):
                add_group(g, SG)
                print('LOOP ITERATION {}: ADD GROUP {} TO SG'.format(loop_iteration, g.index))

                remove_group(g, UG)
                print('LOOP ITERATION {}: REMOVE GROUP {} FROM UG BECAUSE IT IS NOW A SAFE GROUP'.format(loop_iteration, g.index))

            print('LOOP ITERATION {}: CHECK SAFE GROUPS - SelG: {} - g: {}'.format(loop_iteration, is_safe_group(SelG), is_safe_group(g)))

            # Handle which group next? If there is any unsafe group in the pair, continue with it
            if is_unsafe_group(SelG):
                pass    # Keep handling SelG
            elif is_unsafe_group(g):    # Continue with g
                SelG = g
                remove_group(g, UG)
            else:
                # The next iteration we will choose another group to process
                SelG = None

    print('TOTAL LOOPS: {}. UG length: {}. SG length: {}. UM length: {}\n'.format(loop_iteration, len(UG), len(SG), len(UM)))    
    print('TOTAL NUMBER OF TUPLES IN SAFE GROUPS: {}'.format(sum(group_length(group) for group in GROUPS if group_length(group) >= DESIRED_K)))
    print('TOTAL NUMBER OF TUPLES IN SAFE GROUPS 2: {}'.format(sum(group_length(group) for group in SG)))
    print('TOTAL NUMBER OF TUPLES IN UNSAFE GROUPS: {}'.format(sum(group_length(group) for group in GROUPS if group_length(group) < DESIRED_K and group not in UM)))
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
    print('TOTAL NUMBER OF TUPLES IN SAFE GROUPS: {}'.format(sum(group_length(group) for group in GROUPS if group_length(group) >= DESIRED_K)))
    print('TOTAL NUMBER OF TUPLES IN SAFE GROUPS 2: {}'.format(sum(group_length(group) for group in SG)))
    print('TOTAL NUMBER OF TUPLES IN UNSAFE GROUPS: {}'.format(sum(group_length(group) for group in GROUPS if group_length(group) < DESIRED_K)))
    print('TOTAL NUMBER OF TUPLES IN UNSAFE GROUPS 2: {}'.format(sum(group_length(group) for group in UG)))
    print('TOTAL NUMBER OF TUPLES IN UM: {}'.format(sum(group_length(group) for group in UM)))    

    eval_results(R_initial, GROUPS, output_file_name, start_time)


if __name__ == '__main__':
    # sys.stdout = open("log/m3ar_results.log", "w")
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
    output_file_name = 'out_m3ar_k_' + str(DESIRED_K) + '_' + data_file_path.split('/')[-1].split('.')[0] + '.data'
    m3ar_algo(D, R_initial, output_file_name)

    sys.stdout.close()
