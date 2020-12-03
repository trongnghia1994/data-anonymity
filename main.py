import csv
import pandas
import math
import random
import time
from dataclasses import dataclass
from apyori import apriori
from mlxtend.frequent_patterns import apriori, fpmax, fpgrowth
from mlxtend.preprocessing import TransactionEncoder
from collections import namedtuple
from itertools import combinations

DATA_FILE_PATH = './dataset/adult.data'
DATA_COLUMNS = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
RETAINED_DATA_COLUMNS = ['age', 'sex', 'marital-status', 'native-country',
                         'race', 'education', 'hours-per-week', 'capital-gain', 'workclass']
QUASI_ATTRIBUTES = RETAINED_DATA_COLUMNS[:6]
MIN_SUP = 0.0045
MIN_CONF = 0.2
MIN_LENGTH = 2
DESIRED_K = 10

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
    data: list
    group_index: int


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


def rule_budget(a_rule):    # Budget of the rule A -> B where A, B are sets of attribute values. The smaller the budget the more risk it will be lost
    if not item_set_contains_quasi_attr(a_rule.B):
        return min(a_rule.support - MIN_SUP, a_rule.support*(a_rule.confidence - MIN_CONF) / a_rule.confidence*(1 - MIN_CONF))
    else:
        return min(a_rule.support - MIN_SUP, a_rule.support*(a_rule.confidence - MIN_CONF) / a_rule.confidence)


def is_safe_group(a_group: GROUP):
    return group_length(a_group) >= DESIRED_K


def group_length(a_group: GROUP):
    return (len(a_group.origin_tuples) + len(a_group.received_tuples))


def build_groups(dataset: pandas.DataFrame, quasi_attrs: list = QUASI_ATTRIBUTES):
    UG, SG = [], []
    GROUPS = dataset.groupby(quasi_attrs)
    group_index = 0
    for _, df_group in GROUPS:
        group_data = []
        for row in df_group.iterrows():
            data_tuple = DATA_TUPLE(row, group_index)
            group_data.append(data_tuple)

        group = GROUP(group_index, len(group_data), group_data, [])
        if is_safe_group(group):
            SG.append(group)
        else:
            UG.append(group)

        group_index += 1

    return SG, UG


# Construct the rule set we care (relating to quasi attributes)
def construct_r_care(R_initial: list):
    return [rule for rule in R_initial if rule_contains_quasi_attr(rule)]


# Check if a data tuple supports a rule
def data_tuple_supports_a_rule(data_tuple: pandas.Series, rule: RULE):
    rule_items = rule.A + rule.B
    for item in rule_items:
        if data_tuple.get(item.attr) != item.value:
            return False

    return True


# Construct the rule set affected by a migration operation of tuples T from group i to group j
def construct_r_affected_by_a_migration(R_care: list, T: list):
    R_result = set()
    for rule in R_care:
        for data_tuple in T:
            if data_tuple_supports_a_rule(data_tuple, r):
                R_result.add(rule)

    return list(R_result)


# POLICIES
'''
1. A k-unsafe group once has received tuple(s), it can only
continue receiving tuple(s); otherwise, as its tuple(s) migrate
to another group, it can only continue giving its tuple(s) to
other groups. The policy does not apply to k-safe groups. ONLY for k-safe groups
'''


def has_group_received_tuples(a_group: GROUP):
    return len(a_group.received_tuples) > 0


def has_group_given_tuples(a_group: GROUP):
    return len(a_group.origin_tuples) > a_group.origin_len

def choose_tuples_for_migration(R_care: list, a_group: GROUP, no_tuples: int):  # Return indices of elements selected
    current_group_tuples = a_group.origin_tuples + a_group.received_tuples
    comb = combinations(len(range(current_group_tuples)), no_tuples)

    for a_comb in list(comb):   # Encounter a combination that is good get it
        if migration_check_budget_all_rules_affected_positive(R_care, [current_group_tuples[i] for i in a_comb]):
            return a_comb

    return None

'''
2. g i⎯⎯T→g j and ∀t ∈T → (∀r ∈ Rt,gi →g j → Budget r > 0)
'''


def migration_check_budget_all_rules_affected_positive(R_care, T):
    R_affected = construct_r_affected_by_a_migration(R_care, T)
    return all(rule.budget > 0 for rule in R_affected)


'''
3. Calculate number of migrant tuples in a migration operation
'''


# Migrate tuples from group i to group j
def cal_number_of_migrant_tuples(group_i, group_j, migration_direction='l2r'):
    if not is_safe_group(group_j):
        if migration_direction == 'l2r':
            return min(group_length(group_i), DESIRED_K - group_length(group_j))
        else:
            return min(group_length(group_j), DESIRED_K - group_length(group_i))
    else:
        if migration_direction == 'l2r':
            return group_length(group_i)
        else:
            no_migrant_tuples = min(DESIRED_K - group_length(group_i),
                                    group_length(group_j) - DESIRED_K, len(group_j.origin_tuples))
            if no_migrant_tuples == 0:
                print('PROBLEM: Cannot perform this move')
            return 0


# END OF POLICIES
def apply_policies(R_care, group_i, group_j):   # Consider a member migration from group i to group j
    no_migrant_tuples, risk_reduction, time_elapsed = -1, -1, -1
    start = time.time()
    # Policy 1
    if not is_safe_group(group_i) and has_group_received_tuples(group_i):
        return no_migrant_tuples, risk_reduction, time_elapsed
    if not is_safe_group(group_j) and has_group_given_tuples(group_j):  # Group j cannot receive more tuples because it is an unsafe group and it has given its tuples to another before
        return no_migrant_tuples, risk_reduction, time_elapsed
    # Policy 3
    no_migrant_tuples = cal_number_of_migrant_tuples(group_i, group_j, migration_direction='l2r')
    # Policy 2
    if no_migrant_tuples > 0:
        # Select tuples from group i to satisfy the condition in which the budget all rules affected greater than 0
        current_group_tuples = group_i.origin_tuples + group_i.received_tuples
        for i in choose_tuples_for_migration(R_care, group_i, no_migrant_tuples):   # Move tuples to another group
            new_tuples = # TODO: Continue to write
            group_i.origin_tuples = 
            group_j.received_tuples.append(current_group_tuples[i])
            tuples_to_move = [current_group_tuples[i] for i in choose_tuples_for_migration(R_care, group_i, no_migrant_tuples)]
        group_i.origin_tuples.remove
        group_j.received_tuples.append(tuples_to_move)


    time_elapsed = time.time() - start
    return no_migrant_tuples, risk_reduction, time_elapsed

# Apply policies to perform a useful migration operation
def find_group_to_migrate(selected_group: GROUP, UG, SG):
    remaining_groups = UG + SG
    for group in remaining_groups:
        # Start to apply policies
        effective_factor = apply_policies(selected_group, group)
        effective_factor_reverse = apply_policies(group, selected_group)


def disperse(a_group):  # Disperse, return or move (Giai tan) tuples that have been migrated into this group to another one
    pass

def m3ar_algo(D, R_initial):
    # Build groups from the dataset then split G into 2 sets of groups: safe groups SG and unsafe groups UG
    SG, UG = build_groups(D)
    print(len(SG), len(UG))
    print(type(SG[0].origin_tuples[0].data))
    # for row in SG[0].iterrows():    # Check if data tuples in a group supports a rule
    #     print('DEBUG', data_tuple_supports_a_rule(row[1], R_initial[0]))
    UM = set()  # Set of groups that cannot migrate member with other groups
    print('Number of safe groups and unsafe groups', len(SG), len(UG))
    R_care = construct_r_care(R_initial)  # List of cared rules
    print('R care', R_care)
    for r in R_care:
        r.budget = rule_budget(r)
    print('R care now', R_care)
    SelG = None
    # while (len(UG) > 0) or (SelG):
    #     if (not SelG):  # Randomly pick a group SelG from unsafe groups set
    #         SelG = random.choice(UG)
    #         UG.remove(SelG)

    #     # Find the most appropriate group g in UG and SG to perform migration with SelG
    #     most_useful_g = find_group_to_migrate_with_a_group(SelG, UG, SG)
    #     if not most_useful_g:   # If cannot find such a group, add it to the unmigrant UM set
    #         UM.add(SelG)
    #     else:
    #         pass # Do not know???

    #     if most_useful_g in UG:
    #         UG.remove(most_useful_g)

    #     # Handle SelG next?
    #     if not is_safe_group(SelG):
    #         pass    # Keep handling SelG
    #     elif not is_safe_group(most_useful_g):
    #         SelG = most_useful_g
    #     else:
    #         SelG = None # The next iteration we will choose another group SelG to process

    # if len(UM) > 0:
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
