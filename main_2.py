import csv
import pandas
import math
import random
import time
import traceback
from dataclasses import dataclass
from itertools import combinations
from common import *


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


def is_safe_group(a_group: GROUP, k=DESIRED_K):
    return group_length(a_group) >= k


def is_unsafe_group(a_group: GROUP, k=DESIRED_K):
    return not is_safe_group(a_group, k) and group_length(a_group) > 0


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

    # MODIFICATION 1: Sort groups in UG by group length descendingly
    UG = sorted(UG, key=lambda gr: group_length(gr), reverse=True)

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
        group_risk_before = group_risk(group_i) + group_risk(group_j)
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

def export_dataset(groups: list):
    '''Write the modified dataset to file'''
    def write_data_tuple(t: DATA_TUPLE, f):
        str_concat = ''
        for index, value in t.data.items():
            str_concat += str(value)
            str_concat += ','

        str_concat = str_concat[:-1]
        f.write(str_concat + '\n')
        
    output_file_name = 'modified_ds.data'
    with open(output_file_name, 'w') as f:
        for group in groups:
            for t in group.origin_tuples:
                write_data_tuple(t, f)

            for t in group.received_tuples:
                write_data_tuple(t, f)
                        

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


def m3ar_modified_algo(D, R_initial):
    start_time = time.time()
    # First construct R care
    R_care = construct_r_care(R_initial)
    for r in R_care:
        r.budget = rule_budget(r)
    print('R care', R_care)
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

            no_tuples_needed_to_be_a_safe_group = DESIRED_K - group_length(SelG)
            no_tuples_picked = 0
            for data_tuple in free_tuples:
                R_affected = construct_r_affected_by_a_migration(R_care, [data_tuple], SelG)
                if all_rules_having_positive_budgets(R_affected):
                    for rule in R_affected:
                        rule.budget -= 1
                    # Perform a migration of this free tuple to the destination group               
                    SelG.received_tuples.append(data_tuple)
                    convert_quasi_attributes(data_tuple, group_first_tuple(SelG))
                    print('Data tuple picked is from group {}'.format(data_tuple.group_index))
                    source_group = find_group(data_tuple.group_index, GROUPS)
                    source_group.origin_tuples.remove(data_tuple)
                    free_tuples.remove(data_tuple)
                    no_tuples_picked += 1
                    if no_tuples_picked == no_tuples_needed_to_be_a_safe_group:
                        # Pick enough, break to process the next unsafe group                        
                        add_group(SelG, SG)
                        remove_group(SelG, UG)
                        print('Number of safe groups now is: {}. Number of free tuples is: {}'.format(len(SG), len(free_tuples)))
                        SelG = None
                        break

            # If look up in all free tuples but cannot find enough tuples to make this group safe, return them to source group
            if no_tuples_picked < no_tuples_needed_to_be_a_safe_group:                
                for data_tuple in SelG.received_tuples:
                    SelG.received_tuples.remove(data_tuple)
                    source_group = find_group(data_tuple.group_index, GROUPS)
                    source_group.origin_tuples.append(data_tuple)
                    convert_quasi_attributes(data_tuple, group_first_tuple(source_group))
                    R_affected = construct_r_affected_by_a_migration(R_care, [data_tuple], source_group)
                    free_tuples.append(data_tuple)
                    add_group(SelG, UG)
                    remove_group(SelG, SG)
                    for rule in R_affected:
                        rule.budget += 1

                SelG = None

    total_time = time.time() - start_time
    print('STAGE 1 RUN TIME: {} seconds'.format(total_time))
    print('TOTAL LOOPS: {}'.format(loop_iteration))
    print('NUMBER OF UNSAFE GROUPS AND SAFE GROUPS AFTER STAGE 1: {}, {}'.format(len(UG), len(SG)))
    print('NUMBER OF FREE TUPLES: {}'.format(len(free_tuples)))
    # pprint_groups(GROUPS)
    export_dataset(GROUPS)

    # STAGE 2
    

    print('==FINAL RULES==')
    for rule in R_care:
        print(rule)
    print('=========FINAL GROUPS=========')
    pprint_groups(GROUPS)
    export_dataset(GROUPS)


# Main
# A dataset reaches k-anonymity if total risks of all groups equals to 0
# A Member Migration operation g(i)-T-g(j) is valuable when the risk of data is decreased after performing that Member Migration operation.
D = pandas.read_csv(DATA_FILE_PATH, names=DATA_COLUMNS,
                    index_col=False, skipinitialspace=True)
D = D[RETAINED_DATA_COLUMNS]
dataset_length = D.shape[0]
print('Dataset length', dataset_length)
MIN_SUP = MIN_SUP * dataset_length
R_initial = [RULE([RULE_ITEM('Male', 'sex')], [RULE_ITEM(
    'White', 'race')], support=0.62, confidence=0.8378378378378378, budget=0)]
# Convert support percentage to support count
for rule in R_initial:
    rule.support = int(rule.support * dataset_length)
print(R_initial)
m3ar_modified_algo(D, R_initial)
