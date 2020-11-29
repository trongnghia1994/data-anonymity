import csv, pandas, math, random
from apyori import apriori
from mlxtend.frequent_patterns import apriori, fpmax, fpgrowth
from mlxtend.preprocessing import TransactionEncoder

DATA_FILE_PATH = './dataset/adult.data'
DATA_COLUMNS = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
RETAINED_DATA_COLUMNS = ['age', 'sex', 'marital-status', 'native-country', 'race', 'education', 'hours-per-week', 'capital-gain', 'workclass']
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


def item_set_contains_quasi_attr(item_set):
    pass

def group_risk(group_length):
    if group_length >= DESIRED_K:
        return 0
    else:
        return 2*DESIRED_K - group_length

def rule_budget(a_rule):    # Budget of the rule A -> B where A, B are sets of attribute values. The smaller the budget the more risk it will be lost    
    if not item_set_contains_quasi_attr(a_rule.B):
        return min(a_rule.support - MIN_SUP, math.floor(a_rule.support*(a_rule.confidence - MIN_CONF) / a_rule.confidence*(1 - MIN_CONF) ))
    else:
        return min(a_rule.support - MIN_SUP, math.floor(a_rule.support*(a_rule.confidence - MIN_CONF) / a_rule.confidence ))


def is_safe_group(a_group):
    return len(a_group == DESIRED_K)

def cal_number_of_migrant_tuples(group_i, group_j):   # Migrate tuples from group i to group j
    if not is_safe_group(group_j):  
        return min(len(group_i), DESIRED_K - len(group_j))
    else:
        pass


# A dataset reaches k-anonymity if total risks of all groups equals to 0
# A Member Migration operation g(i)-T-g(j) is valuable when the risk of data is decreased after performing that Member Migration operation.

D = []  # Dataset

import csv

# with open(DATA_FILE_PATH, newline='') as f:
#     reader = csv.reader(f)
#     D = list(reader)

D = pandas.read_csv(DATA_FILE_PATH, names=DATA_COLUMNS, index_col=False)
D = D[RETAINED_DATA_COLUMNS]
GROUPS = D.groupby(QUASI_ATTRIBUTES)
i = 0 
for key, item in GROUPS:
    a_group = GROUPS.get_group(key)
    if len(a_group) > 1:
        print(a_group)
        i += 1
        if i == 10:
            break

# print(GROUPS)

def build_groups(dataset):
    return set()

def split(groups):
    return set(), set()

def construct_r_care():
    return set()

def find_group_to_migrate(UG, SG):
    return None

def disperse(a_group):  # Disperse, return or move (Giai tan) tuples that have been migrated into this group to another one
    pass

def m3ar_algo(D, R):
    G = build_groups(D) # Build groups from the dataset
    SG, UG = split(G)   # Split G into 2 sets of groups safe groups SG and unsafe groups UG
    UM = set()  # Set of groups that cannot migrate member with other groups
    R_care = construct_r_care() # List of cared rules
    R_budgets = [rule_budget(r) for r in R_care]
    SelG = None
    while (len(UG) > 0) or (SelG):
        if (not SelG):  # Randomly pick a group SelG from unsafe groups set
            SelG = random.sample(UG, 1)[0]
            UG.remove(SelG)

        # Find the most appropriate group g in UG and SG to perform migration with SelG
        most_useful_g = find_group_to_migrate(UG, SG)
        if not most_useful_g:   # If cannot find such a group, add it to the unmigrant UM set
            UM.add(SelG)
        else:
            pass # Do not know???

        if most_useful_g in UG:
            UG.remove(most_useful_g)

        # Handle SelG next?
        if not is_safe_group(SelG):
            pass    # Keep handling SelG
        elif not is_safe_group(most_useful_g):
            SelG = most_useful_g
        else:
            SelG = None # The next iteration we will choose another group SelG to process

    if len(UM) > 0:
        for g in UM:
            disperse(g)
