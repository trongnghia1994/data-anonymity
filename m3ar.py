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


sys.stdout = open("log/m3ar_results.log", "w")


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


def is_safe_group(a_group: GROUP):
    return group_length(a_group) >= DESIRED_K


def is_unsafe_group(a_group: GROUP):
    return not is_safe_group(a_group)


def group_length(a_group: GROUP):
    return (len(a_group.origin_tuples) + len(a_group.received_tuples))


def calc_risk_reduction(group_i: GROUP, group_j: GROUP, no_migrant_tuples: int):
    '''Calculate the risk reduction in case performing 
    a migration operation of <no_migrant_tuples> from group i to group j'''
    risk_before = group_risk(group_i) + group_risk(group_j)
    group_i_length_after = group_length(group_i) - no_migrant_tuples
    group_i_risk_after = 0 if group_i_length_after >= DESIRED_K else 2*DESIRED_K - group_i_length_after
    group_j_length_after = group_length(group_j) + no_migrant_tuples
    group_j_risk_after = 0 if group_j_length_after >= DESIRED_K else 2*DESIRED_K - group_j_length_after
    return risk_before - (group_i_risk_after + group_j_risk_after)


def build_groups(dataset: pandas.DataFrame, quasi_attrs: list = QUASI_ATTRIBUTES):
    '''Build safe groups and unsafe groups from the initial dataset'''
    UG, SG = [], []
    DF_GROUPS = dataset.groupby(quasi_attrs)
    group_index = 0
    for _, df_group in DF_GROUPS:
        group_data = []
        for row in df_group.iterrows():
            index, data = row
            data_tuple = DATA_TUPLE(index, data, group_index)
            group_data.append(data_tuple)

        group = GROUP(group_index, len(group_data), group_data, [])
        if is_safe_group(group):
            SG.append(group)
        else:
            UG.append(group)

        group_index += 1

    GROUPS = SG + UG
    return GROUPS, SG, UG


# Construct the rule set we care (relating to quasi attributes)
def construct_r_care(R_initial: list):
    return [rule for rule in R_initial if rule_contains_quasi_attr(rule)]


def data_tuple_supports_a_rule(data_tuple: DATA_TUPLE, rule: RULE):
    '''Check if a data tuple supports a rule'''
    rule_items = rule.A + rule.B
    for item in rule_items:
        if data_tuple.data.get(item.attr) != item.value:
            # Compare attribute value of data tuple with rule attribute value
            return False

    return True


def rule_contains_attr_val(rule: RULE, attr_name, attr_value):
    '''Check if a rule contains the attribute value indicated by attr_name, attr_value'''
    rule_items = rule.A + rule.B
    for rule_item in rule_items:
        if rule_item.attr == attr_name and rule_item.value != attr_value:
            return True

    return True


def group_first_tuple(a_group: GROUP):
    '''Return the first tuple of group from origin_tuples or received_tuples'''
    if len(a_group.origin_tuples) > 0:
        return a_group.origin_tuples[0]

    if len(a_group.received_tuples) > 0:
        return a_group.origin_tuples[0]

    return None


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


def convert_quasi_attributes(data_tuple: DATA_TUPLE, dst_data_tuple: DATA_TUPLE):
    '''Convert all quasi attributes' values of <data_tuple> 
    to the corresponding of <dst_data_tuple>'''
    data_tuple.data.update(dst_data_tuple.data[:6])


def do_migration(group_i: GROUP, group_j: GROUP, migrant_tuples_indices: list):
    '''Migrate tuples from group i to group j. Migrant tuples' indices figured out in migrant indices'''
    kept_tuples_indices = list(set(range(len(group_i.origin_tuples))) - set(migrant_tuples_indices)) 
    dst_group_first_tuple = group_first_tuple(group_j)
    # Move tuples to another group
    for i in migrant_tuples_indices:
        tuple_to_move = group_i.origin_tuples[i]
        convert_quasi_attributes(tuple_to_move, dst_group_first_tuple)
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


GROUPS, SG, UG, UM = [], [], [], []


def disperse(R_care: list, a_group: GROUP):
    '''
    This group is still unsafe and it contains tuples migrated from other groups
    In loop we cannot find another group to perform migration with this group
    Disperse, return or move (Giai tan) tuples that have been migrated into this group to other ones
    '''
    print('DISPERSE GROUP', a_group.index)
    pprint_groups([a_group])
    # For all receiving tuples, return them to the source group
    for data_tuple in a_group.received_tuples:
        source_group = find_group(data_tuple.group_index, GROUPS)
        source_group.origin_tuples.append(data_tuple)
        print('DISPERSE: GROUP {} RETAKES THE DATA TUPLE {}'.format(source_group.index, data_tuple.index))
        R_affected = construct_r_affected_by_a_migration(R_care, [data_tuple], source_group)
        for rule in R_affected:
            rule.budget += 1
        # If the source group now has only 1 tuple, add it to UM to be processed
        if group_length(source_group) == 1 and source_group not in UM:
            UM.append(source_group)

    for data_tuple in a_group.origin_tuples:
        # Find the most appropriate group g in SG to perform migration
        dst_group_to_migrate = find_group_to_move_dispersing(R_care, data_tuple, SG)
        if dst_group_to_migrate:
            print('DISPERSE: GROUP {} TAKES THE DATA TUPLE {} FROM GROUP {}'.format(dst_group_to_migrate.index, data_tuple.index, a_group.index))
            dst_group_to_migrate.received_tuples.append(data_tuple)


def pprint_data_tuple(data_tuple: DATA_TUPLE):
    str_concat = '{}: '.format(data_tuple.index)
    for index, value in data_tuple.data.items():
        str_concat += str(value)
        str_concat += ','
    print(str_concat)


def pprint_groups(groups: list):
    for group in groups:
        print('================================')
        print('Group index', group.index)
        print('Group length:', len(group.origin_tuples) + len(group.received_tuples), '===== Is safe?', is_safe_group(group))
        print('Group origin tuples:', 'Empty' if len(group.origin_tuples) == 0 else '')
        for t in group.origin_tuples:
            pprint_data_tuple(t)
        print('Group received tuples:', 'Empty' if len(group.received_tuples) == 0 else '')
        for t in group.received_tuples:
            pprint_data_tuple(t)
        print('================================')

def export_dataset(groups: list, output_file_name='m3ar_ds.data'):
    '''Write the modified dataset to file'''
    def write_data_tuple(t: DATA_TUPLE, f):
        str_concat = ''
        for index, value in t.data.items():
            str_concat += str(value)
            str_concat += ','

        str_concat = str_concat[:-1]
        f.write(str_concat + '\n')
        
    output_file_name = 'output/' + output_file_name
    with open(output_file_name, 'w') as f:
        for group in groups:
            for t in group.origin_tuples:
                write_data_tuple(t, f)

            for t in group.received_tuples:
                write_data_tuple(t, f)


def m3ar_algo(D, R_initial, output_file_name):
    start_time = time.time()
    # Build groups from the dataset then split G into 2 sets of groups: safe groups SG and unsafe groups UG
    GROUPS, SG, UG = build_groups(D)
    UM = []  # Set of groups that cannot migrate member with other groups
    print('K =', DESIRED_K)
    print('Number of safe groups and unsafe groups:', len(SG), len(UG))
    R_care = construct_r_care(R_initial)  # List of cared rules
    for r in R_care:
        r.budget = rule_budget(r)
    print('R care', R_care)
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
            else:
                # The next iteration we will choose another group to process
                SelG = None

    total_time = time.time() - start_time
    print('RUN TIME: {} seconds'.format(total_time))
    print('TOTAL LOOPS: {}. UG length: {}. SG length: {}. UM length: {}\n'.format(loop_iteration, len(UG), len(SG), len(UM)))    
    print('=======================================')
    print('=======================================')
    print('=======================================')
    print('START TO DISPERSE', len(UM), 'UM GROUPS')
    print('NUMBER OF UM GROUPS WITH LENGTH > 0:', sum(1 for group in UM if group_length(group) > 0))
    if len(UM) > 0: # Disperse
        for g_um in UM:
            if group_length(g_um) > 0:  # Just consider group with length > 0
                disperse(R_care, g_um)

    print('AFTER DISPERSING: NUMBER OF UM GROUPS WITH LENGTH > 0:', sum(1 for group in UM if group_length(group) > 0))
    print('TOTAL NUMBER OF TUPLES IN SAFE GROUPS: {}'.format(sum(group_length(group) for group in SG)))
    print('TOTAL NUMBER OF TUPLES IN UNSAFE GROUPS: {}'.format(sum(group_length(group) for group in UG)))

    print('==FINAL RULES==')
    for rule in R_care:
        print(rule)
    print('=========FINAL GROUPS=========')
    pprint_groups(GROUPS)
    export_dataset(GROUPS, output_file_name)


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
    output_file_name = 'out_m3ar_k_' + str(DESIRED_K) + '_' + data_file_path.split('/')[-1].split('.')[0] + '.data'
    m3ar_algo(D, R_initial, output_file_name)

    sys.stdout.close()
