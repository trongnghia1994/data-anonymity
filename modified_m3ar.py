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
from ar_mining import cal_supp_conf
from common import *


sys.stdout = open("log/modified_algo_results.log", "w")


OUTPUT_DATASET_PATH = 'modified_ds.data'


# Check if an itemset contains quasi attributes
def item_set_contains_quasi_attr(item_set: list):
    return any(rule_item.attr in QUASI_ATTRIBUTES for rule_item in item_set)


# Check if an itemset contains quasi attributes
def rule_contains_quasi_attr(rule: RULE):
    return any(item_set_contains_quasi_attr(item_set) for item_set in [rule.A, rule.B])


def group_risk(a_group):
    group_len = group_length(a_group)
    if group_len >= DESIRED_K:
        return 0
    else:
        return 2*DESIRED_K - group_len


def rule_budget(a_rule):
    '''Budget of the rule A -> B where A, B are sets of attribute values. The smaller the budget the more risk it will be lost'''    
    if not item_set_contains_quasi_attr(a_rule.B):
        # If the right hand side of the rule does not contain any quasi attribute
        return min(a_rule.support - MIN_SUP, int(a_rule.support*(a_rule.confidence - MIN_CONF) / a_rule.confidence*(1 - MIN_CONF)))
    else:
        return min(a_rule.support - MIN_SUP, int(a_rule.support*(a_rule.confidence - MIN_CONF) / a_rule.confidence))


def calc_risk_reduction(group_i: GROUP, group_j: GROUP, no_migrant_tuples: int):
    '''Calculate the risk reduction in case performing 
    a migration operation of <no_migrant_tuples> from group i to group j'''
    risk_before = group_risk(group_i) + group_risk(group_j)
    group_i_length_after = group_length(group_i) - no_migrant_tuples
    group_i_risk_after = 0 if group_i_length_after >= DESIRED_K else 2*DESIRED_K - group_i_length_after
    group_j_length_after = group_length(group_j) + no_migrant_tuples
    group_j_risk_after = 0 if group_j_length_after >= DESIRED_K else 2*DESIRED_K - group_j_length_after
    return risk_before - (group_i_risk_after + group_j_risk_after)


# Construct the rule set we care (relating to quasi attributes)
def construct_r_care(R_initial: list):
    return [rule for rule in R_initial if rule_contains_quasi_attr(rule)]


def rule_contains_attr_val(rule: RULE, attr_name, attr_value):
    '''Check if a rule contains the attribute value indicated by attr_name, attr_value'''
    rule_items = rule.A + rule.B
    for rule_item in rule_items:
        if rule_item.attr == attr_name and rule_item.value != attr_value:
            return True

    return True


def move_data_tuple_affect_a_rule(data_tuple: DATA_TUPLE, rule: RULE, group_j: GROUP):
    group_j_first_tuple = group_first_tuple(group_j)
    # Loop through quasi attributes of the data tuple then compare with the destination group (group j)'s first tuple
    for attr in QUASI_ATTRIBUTES:
        if group_j_first_tuple.data.get(attr) != data_tuple.data.get(attr):
            if rule_contains_attr_val(rule, attr, data_tuple.data.get(attr)):
                return True

    return False


def construct_r_affected_by_a_migration(R_care: list, T: list, group_j: GROUP):
    '''Construct the rule set affected 
    by a migration operation of tuples T from group i to group j'''
    R_result = []    
    for rule in R_care:
        for data_tuple in T:
            if move_data_tuple_affect_a_rule(data_tuple, rule, group_j):
                R_result.append(rule)

    return R_result


# POLICIES
'''
1. A k-unsafe group once has received tuple(s), it can only
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
2. g(i)⎯T->g(j) and ∀t∈T → (∀r ∈ R(t),g(i)->g(j)->Budget(r) > 0)
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
3. Calculate number of migrant tuples in a migration operation
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
    for group in remaining_groups:
        if group.index != selected_group.index:
            # Start to apply policies
            factors = apply_policies(R_care, selected_group, group, 'l2r')
            factors_reverse = apply_policies(R_care, group, selected_group, 'r2l')
            if factors[0]:
                results.append(factors)
            if factors_reverse[0]:
                results.append(factors_reverse)

    if len(results) == 0:
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
                        

def cal_number_of_free_tuples(groups: list, k: int):
    if not k:
        K_SET = list(range(5, 30, 5))
    else:
        K_SET = [k]
    for k in K_SET:
        no_free_tuples = 0
        print('WITH K={}'.format(k))
        for group in groups:
            no_free_tuples += group_length(group) - k if group_length(group) >= k else 0
        print('Number of free tuples: {}'.format(no_free_tuples))


def generate_free_tuples(SG: list):
    '''
    Pick free tuples in each safe group to establish a set of free tuples. Just pick the neccessary tuples at first of each safe group
    '''
    free_tuples = []
    for group in SG:
        group_free_tuples = group.origin_tuples[:group_length(group) - DESIRED_K]
        free_tuples.extend(group_free_tuples)

    return free_tuples


def all_rules_having_positive_budgets(rules:list):
    return all([rule.budget > 0 for rule in rules])


def m3ar_modified_algo(D, R_initial, output_file_name='m3ar_modified.data'):
    start_time = time.time()
    # First construct R care
    R_care = construct_r_care(R_initial)
    for r in R_care:
        r.budget = rule_budget(r)
    print('R care')
    pprint_rule_set(R_care)
    print('===============================================================================')    
    # Build groups from the dataset then split G into 2 sets of groups: safe groups SG and unsafe groups UG
    GROUPS, SG, UG = build_groups(D)
    print('K =', DESIRED_K)
    print('There are {} safe groups and {} unsafe groups'.format(len(SG), len(UG)))
    cal_number_of_free_tuples(GROUPS, DESIRED_K)
    free_tuples = generate_free_tuples(SG)    
    SelG = None
    loop_iteration = 0
    # STAGE 1: Pick free tuples into unsafe groups
    while (len(UG) > 0) and len(free_tuples) > 0 and all_rules_having_positive_budgets(R_care):
        loop_iteration += 1
        if SelG is None:
            # MODIFICATION 1: UG was already sorted by group length. Process the UG groups one by one, start with the unsafe group with the longest length
            # The longer the group length, the more chance it will soon become a safe group
            SelG = UG.pop(0)
            print('LOOP ITERATION {}. Pop the unsafe group with the longest length. SelG index: {}. SelG length: {}'.format(loop_iteration, SelG.index, group_length(SelG)))
            print('Rules budget now are:')
            print([rule.budget for rule in R_care])

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

    total_time = time.time() - start_time
    print('TOTAL LOOPS: {}'.format(loop_iteration))
    print('AFTER STAGE 1')
    print('TOTAL NUMBER OF SAFE GROUPS: {}'.format(len([group for group in GROUPS if group_length(group) >= DESIRED_K])))
    print('TOTAL NUMBER OF UNSAFE GROUPS: {}'.format(len([group for group in GROUPS if 0 < group_length(group) < DESIRED_K])))
    print('NUMBER OF WRONG SAFE GROUPS: {}'.format(len([g for g in SG if group_length(g) < DESIRED_K])))
    print('NUMBER OF UNSAFE GROUPS AND SAFE GROUPS: {}, {}'.format(len(UG), len(SG)))
    print('TOTAL NUMBER OF TUPLES IN SAFE GROUPS: {}'.format(sum(group_length(group) for group in GROUPS if group_length(group) >= DESIRED_K)))
    print('TOTAL NUMBER OF TUPLES IN UNSAFE GROUPS: {}'.format(sum(group_length(group) for group in GROUPS if group_length(group) < DESIRED_K)))
    print('NUMBER OF FREE TUPLES: {}'.format(len(free_tuples)))
    # pprint_groups(GROUPS)
    # export_dataset(GROUPS)

    # STAGE 2: PROCESS ONE BY ONE IN THE UNSAFE GROUP, START WITH GROUP WITH SHORT LENGTH
    UG = sorted(UG, key=lambda gr: group_length(gr))
    UG_SMALL_DISPERSED, UG_BIG_DISPERSED = [], []
    for unsafe_group in UG:
        if group_length(unsafe_group) <= DESIRED_K / 2:  # With small group disperse them
            first_tuple_of_this_unsafe_group = unsafe_group.origin_tuples[0]
            dst_group = find_group_to_move_dispersing(R_care, first_tuple_of_this_unsafe_group, SG)
            if dst_group:
                R_affected = construct_r_affected_by_a_migration(R_care, unsafe_group.origin_tuples, dst_group)
                for rule in R_affected:
                    rule.budget -= 1
                print('BEFORE DISPERSE SMALL GROUP', unsafe_group.index, group_length(unsafe_group))
                for data_tuple in unsafe_group.origin_tuples:
                    convert_quasi_attributes(data_tuple, dst_group)
                    dst_group.received_tuples.append(data_tuple)                                    
                for data_tuple in unsafe_group.received_tuples:
                    convert_quasi_attributes(data_tuple, dst_group)
                    dst_group.received_tuples.append(data_tuple)
                unsafe_group.origin_tuples = []
                unsafe_group.received_tuples = []
                print('AFTER DISPERSE SMALL GROUP', unsafe_group.index, group_length(unsafe_group))
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
            unsafe_group.received_tuples.extend(picked_tuples)
            add_group(unsafe_group, SG) # Now becomes safe
            for data_tuple in picked_tuples:
                source_group = find_group(data_tuple.group_index, GROUPS)
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

    # STAGE 3: PROCESS ONE BY ONE IN THE REMAINING UNSAFE GROUP, START WITH GROUP WITH SHORT LENGTH
    for unsafe_group in UG:
        first_tuple_of_this_unsafe_group = unsafe_group.origin_tuples[0]
        dst_group = find_group_to_move_dispersing(R_care, first_tuple_of_this_unsafe_group, SG)
        if dst_group:
            R_affected = construct_r_affected_by_a_migration(R_care, unsafe_group.origin_tuples, dst_group)
            for rule in R_affected:
                rule.budget -= 1
            print('BEFORE DISPERSE SMALL GROUP', unsafe_group.index, group_length(unsafe_group))
            for data_tuple in unsafe_group.origin_tuples:
                convert_quasi_attributes(data_tuple, dst_group)
                dst_group.received_tuples.append(data_tuple)                                    
            for data_tuple in unsafe_group.received_tuples:
                convert_quasi_attributes(data_tuple, dst_group)
                dst_group.received_tuples.append(data_tuple)
            unsafe_group.origin_tuples = []
            unsafe_group.received_tuples = []
            print('AFTER DISPERSE SMALL GROUP', unsafe_group.index, group_length(unsafe_group))
            # remove_group(unsafe_group, UG)
            UG_SMALL_DISPERSED.append(unsafe_group)
        else:
            print('Cannot find any group for group {} to disperse to'.format(unsafe_group.index))

    total_time = time.time() - start_time
    print('TOTAL NUMBER OF TUPLES IN SAFE GROUPS: {}'.format(sum(group_length(group) for group in GROUPS if group_length(group) >= DESIRED_K)))
    # print('TOTAL NUMBER OF TUPLES IN SAFE GROUPS 2: {}'.format(sum(group_length(group) for group in SG)))
    print('TOTAL NUMBER OF TUPLES IN UNSAFE GROUPS: {}'.format(sum(group_length(group) for group in GROUPS if group_length(group) < DESIRED_K)))
    # print('TOTAL NUMBER OF TUPLES IN UNSAFE GROUPS 2: {}'.format(sum(group_length(group) for group in UG)))
    print('RUN TIME: {} seconds'.format(total_time))
    print('=========FINAL GROUPS=========')
    pprint_groups(GROUPS)
    output_file_name = 'output/' + output_file_name
    export_dataset(GROUPS, output_file_name)
    print('==ORIGIN RULES==')
    for rule in R_care:
        pprint_rule(rule)
    # Recalculate support and confidence of rules
    print('==RULES MINED ON MODIFIED DATASET==')
    modified_R_care = cal_supp_conf(output_file_name, RETAINED_DATA_COLUMNS, R_care)
    for rule in modified_R_care:
        pprint_rule(rule)
    print('=========METRICS=========')
    print('Number of groups:', len(GROUPS))
    print('CAVG:', metrics_cavg(GROUPS))


if __name__ == '__main__':
    if len(sys.argv) > 3:
        data_file_path, initial_rules_path, DESIRED_K = sys.argv[1], sys.argv[2], int(sys.argv[3])
    else:
        data_file_path = DATA_FILE_PATH
        initial_rules_path = 'initial_rules.data'
    # A dataset reaches k-anonymity if total risks of all groups equals to 0
    # A Member Migration operation g(i)-T-g(j) is valuable when the risk of data is decreased after performing that Member Migration operation.
    D = pandas.read_csv(data_file_path, names=RETAINED_DATA_COLUMNS, index_col=False, skipinitialspace=True)
    dataset_length = D.shape[0]
    print('Dataset length', dataset_length)    
    MIN_SUP = MIN_SUP * dataset_length
    R_initial = []
    with open(initial_rules_path, 'rb') as f:
        R_initial = pickle.load(f)
    print('R initial', R_initial)
    output_file_name = 'out_mod_m3ar_' + str(DESIRED_K) + '_' + data_file_path.split('/')[-1].split('.')[0] + '.data'
    m3ar_modified_algo(D, R_initial, output_file_name)

    sys.stdout.close()
