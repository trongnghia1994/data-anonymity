import csv
import pandas
import math
import random
import time
from dataclasses import dataclass
from itertools import combinations

DATA_FILE_PATH = './dataset/adult-min.data'
DATA_COLUMNS = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
RETAINED_DATA_COLUMNS = ['age', 'sex', 'marital-status', 'native-country',
                         'race', 'education', 'hours-per-week', 'capital-gain', 'workclass']
QUASI_ATTRIBUTES = RETAINED_DATA_COLUMNS[:6]
MIN_SUP = 0.0045
MIN_CONF = 0.2
MIN_LENGTH = 2
DESIRED_K = 1

'''
    Notions
    Rule A->B
        a_rule:
            A
            B
            support
            confidence
        MIN_SUP sm
        MIN_CONF cm
        Support (A->B) = s
        Confidence (A->B) = c
    Group: Set of tuples
'''


@dataclass
class RULE:
    A: list
    B: list
    support: float
    confidence: float
    budget: float


@dataclass
class RULE_ITEM:
    value: any
    attr: str


@dataclass
class GROUP:
    index: int
    origin_len: int
    origin_tuples: list
    received_tuples: list


@dataclass
class DATA_TUPLE:
    index: int
    data: pandas.Series
    group_index: int


# METRICS FOR DATA QUALITY
def newly_generated_rules_percentage(D):
    pass


def no_loss_rules(D):
    pass


def different_rules_percentage(D):
    pass


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
        return min(a_rule.support - MIN_SUP, a_rule.support*(a_rule.confidence - MIN_CONF) / a_rule.confidence*(1 - MIN_CONF))
    else:
        return min(a_rule.support - MIN_SUP, a_rule.support*(a_rule.confidence - MIN_CONF) / a_rule.confidence)


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
            return False

    return True


def construct_r_affected_by_a_migration(R_care: list, T: list):
    '''Construct the rule set affected 
    by a migration operation of tuples T from group i to group j'''
    R_result = []
    for rule in R_care:
        for data_tuple in T:
            if data_tuple_supports_a_rule(data_tuple, rule):
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


def choose_group_tuples_for_migration(R_care: list, a_group: GROUP, no_tuples: int):  
    '''
    Return indices of tuples selected for a migration  operation of <no_tuples> tuples from <a_group>
    '''
    comb = combinations(range(len(a_group.origin_tuples)), no_tuples)

    for a_comb in list(comb):   # Encounter a combination that is good get it
        if migration_check_budget_all_rules_affected_positive(R_care, [a_group.origin_tuples[i] for i in a_comb]):
            return a_comb

    return None

'''
2. g(i)⎯T->g(j) and ∀t∈T → (∀r ∈ R(t),g(i)->g(j)->Budget(r) > 0)
If migrating tuples T from group i to group j, all rules affected must have positive budgets
'''


def migration_check_budget_all_rules_affected_positive(R_care, T):
    '''Check if budget of the rules affected by a migration operation positive'''
    R_affected = construct_r_affected_by_a_migration(R_care, T)
    return all(rule.budget > 0 for rule in R_affected)


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
    ableToMigrate, migrant_tuples_indices, no_migrant_tuples, risk_reduction, time_elapsed = False, [], -1, -9999, -1
    start = time.time()
    # POLICY 1
    if is_unsafe_group(group_i) and has_group_received_tuples(group_i):
        return ableToMigrate, group_i, group_j, migrant_tuples_indices, no_migrant_tuples, risk_reduction, time_elapsed
    # Group j cannot receive more tuples because it is an unsafe group and it has given its tuples to another before
    if is_unsafe_group(group_j) and has_group_given_tuples(group_j):
        return ableToMigrate, group_i, group_j, migrant_tuples_indices, no_migrant_tuples, risk_reduction, time_elapsed
    # POLICY 3
    no_migrant_tuples = cal_number_of_migrant_tuples(group_i, group_j, migration_direction)
    # POLICY 2
    if no_migrant_tuples > 0:
        group_risk_before = group_risk(group_i) + group_risk(group_j)
        # Select tuples from group i to satisfy the condition in which the budget all rules affected greater than 0
        migrant_tuples_indices = choose_group_tuples_for_migration(R_care, group_i, no_migrant_tuples)
        ableToMigrate = True
        risk_reduction = calc_risk_reduction(group_i, group_j, no_migrant_tuples)

    time_elapsed = time.time() - start    
    return ableToMigrate, group_i, group_j, migrant_tuples_indices, no_migrant_tuples, risk_reduction, time_elapsed


def do_migration(group_i: GROUP, group_j: GROUP, migrant_tuples_indices: list):
    '''Migrate tuples from group i to group j. Migrant tuples' indices figured out in migrant indices'''
    kept_tuples_indices = list(set(range(len(group_i.origin_tuples))) - set(migrant_tuples_indices)) 
    for i in migrant_tuples_indices:   # Move tuples to another group
        group_j.received_tuples.append(group_i.origin_tuples[i])
        tuples_to_move = [group_i.origin_tuples[migrant_i] for migrant_i in migrant_tuples_indices]
    group_i.origin_tuples = [group_i.origin_tuples[kept_i] for kept_i in kept_tuples_indices]
    group_j.received_tuples.extend(tuples_to_move)


# Apply policies to perform a useful migration operation
def find_group_to_migrate(R_care: list, selected_group: GROUP, UG: list, SG: list):
    remaining_groups = UG + SG
    results = []
    for group in remaining_groups:
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
    resultsDataFrame = pandas.DataFrame(results, columns=['ableToMigrate', 'group_i', 'group_j', 'migrant_tuples_indices', 'no_migrant_tuples', 'risk_reduction', 'time_elapsed'])
    # Get the best migration selection
    resultsDataFrame.sort_values(by=['time_elapsed', 'risk_reduction', 'no_migrant_tuples'], ascending=[True, False, True])
    return resultsDataFrame.iloc[0]

def disperse(a_group):
    '''
    This group is still unsafe and it contains tuples migrated from other groups
    In loop we cannot find another group to perform migration with this group
    Disperse, return or move (Giai tan) tuples that have been migrated into this group to other ones
    '''
    pass


def m3ar_algo(D, R_initial):
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
            UG.remove(SelG)

        # Find the most appropriate group g in UG and SG to perform migration with SelG
        result_find_migration = find_group_to_migrate(R_care, SelG, UG, SG)        
        # If cannot find such a group, add it to the unmigrant UM set
        if result_find_migration is None:
            print('LOOP ITERATION {}: NO RESULT FOR MIGRATION'.format(loop_iteration))
            if SelG not in UM:  # Add SelG to UM then process another group
                UM.append(SelG)
                SelG = None
        else:
            print('LOOP ITERATION {}: MIGRATION FROM GROUP {} TO GROUP {}'.format(loop_iteration, result_find_migration.group_i.index, result_find_migration.group_j.index))
            g = result_find_migration.group_i if result_find_migration.group_i.index != SelG.index else result_find_migration.group_j
            # Perform a migration operation
            print('DO MIGRATION')
            do_migration(result_find_migration.group_i, result_find_migration.group_j, result_find_migration.migrant_tuples_indices)

            # Then remove the other group from UG if it is in
            if g in UG:
                UG.remove(g)
                print('REMOVE complete')       

            # Handle SelG next?
            if is_unsafe_group(SelG):
                pass    # Keep handling SelG
            elif is_unsafe_group(g):    # Continue with g
                SelG = g
            else:
                # The next iteration we will choose another group to process
                SelG = None

    print('TOTAL LOOPS: {}. UG length: {}. SG length: {}. UM length: {}\n'.format(loop_iteration, len(UG), len(SG), len(UM)))
    print('=========SG=========')
    print(SG)
    print('=========UG=========')
    print(UG)
    print('=========UM=========')
    print(UM)
    print('=========FINAL GROUPS=========')
    print(GROUPS)
    # if len(UM) > 0: # Disperse
    #     for g in UM:
    #         disperse(g)


# Main
# A dataset reaches k-anonymity if total risks of all groups equals to 0
# A Member Migration operation g(i)-T-g(j) is valuable when the risk of data is decreased after performing that Member Migration operation.
D = pandas.read_csv(DATA_FILE_PATH, names=DATA_COLUMNS,
                    index_col=False, skipinitialspace=True)
D = D[RETAINED_DATA_COLUMNS]
R_initial = [RULE([RULE_ITEM('Male', 'sex')], [RULE_ITEM(
    'White', 'race')], support=0.4, confidence=0.4, budget=0.0)]
m3ar_algo(D, R_initial)
