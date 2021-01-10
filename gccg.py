from common import pick_random_rules, pprint_rule_set, QUASI_ATTRIBUTES, CAT_TREES, RETAINED_DATA_COLUMNS, metrics_cavg, pprint_rule, metrics_cavg_raw, rules_metrics
from ar_mining import cal_supp_conf
from preprocess import preprocess
from Apriori import apriori_gen_rules
from eval import eval_results
from oka_py3.anonymizer import run_oka_with_adult_ds
import pickle, subprocess, time, pandas, sys, time


NUMERIC_ATTRS = {
    'age': 90 - 17,
}

CAT_ATTRS = {
    'marital-status': {
        'depth': 3,
        'data': [
            ['Never-married', '*'],
            ['Married-civ-spouse', 'Married', '*'],
            ['Married-AF-spouse', 'Married', '*'],
            ['Divorced', 'leave', '*'],
            ['Separated', 'leave', '*'],
            ['Widowed', 'alone', '*'],
            ['Married-spouse-absent', 'alone', '*'],
        ],
    },
    'native-country': {
        'depth': 2,
        'data': [
            ['Cambodia','*'],
            ['Canada','*'],
            ['China','*'],
            ['Columbia','*'],
            ['Cuba','*'],
            ['Dominican-Republic','*'],
            ['Ecuador','*'],
            ['El-Salvador','*'],
            ['England','*'],
            ['France','*'],
            ['Germany','*'],
            ['Greece','*'],
            ['Guatemala','*'],
            ['Haiti','*'],
            ['Holand-Netherlands','*'],
            ['Honduras','*'],
            ['Hong','*'],
            ['Hungary','*'],
            ['India','*'],
            ['Iran','*'],
            ['Ireland','*'],
            ['Italy','*'],
            ['Jamaica','*'],
            ['Japan','*'],
            ['Laos','*'],
            ['Mexico','*'],
            ['Nicaragua','*'],
            ['Outlying-US(Guam-USVI-etc)','*'],
            ['Peru','*'],
            ['Philippines','*'],
            ['Poland','*'],
            ['Portugal','*'],
            ['Puerto-Rico','*'],
            ['Scotland','*'],
            ['South','*'],
            ['Taiwan','*'],
            ['Thailand','*'],
            ['Trinadad&Tobago','*'],
            ['United-States','*'],
            ['Vietnam','*'],
            ['Yugoslavia','*'],
        ],
    },
    'race': {
        'depth': 2,
        'data': [
            ['Amer-Indian-Eskimo','*'],
            ['Asian-Pac-Islander','*'],
            ['Black','*'],
            ['Other','*'],
            ['White','*'],
        ],
    },
    'sex': {
        'depth': 2,
        'data': [
            ['Female','*'],
            ['Male','*'],
        ],
    },
    'education': {
        'depth': 2,
        'data': [
            ['10th','*'],
            ['11th','*'],
            ['12th','*'],
            ['1st-4th','*'],
            ['5th-6th','*'],
            ['7th-8th','*'],
            ['9th','*'],
            ['Assoc-acdm','*'],
            ['Assoc-voc','*'],
            ['Bachelors','*'],
            ['Doctorate','*'],
            ['HS-grad','*'],
            ['Masters','*'],
            ['Preschool','*'],
            ['Prof-school','*'],
            ['Some-college','*'],
        ],
    }
}


def intersection(lst1, lst2): 
    lst3 = [value for value in lst1 if value in lst2] 
    return lst3 


def cal_depth_common_ancestor(cat_attr, tuple_1_val, tuple_2_val):
    for branch in CAT_ATTRS[cat_attr]['data']:
        if branch[0] == tuple_1_val:
            tuple_1_val_branch = branch
        if branch[0] == tuple_2_val:
            tuple_2_val_branch = branch

    common_ancestor = intersection(tuple_1_val_branch, tuple_2_val_branch)
    if not common_ancestor:
        raise Exception('Cannot find a common ancestor for {} and {}'.format(tuple_1_val, tuple_2_val))

    common_ancestor = common_ancestor[0]    
    common_ancestor_depth_1 = tuple_1_val_branch.index(common_ancestor) + 1
    common_ancestor_depth_2 = tuple_2_val_branch.index(common_ancestor) + 1
    # print('DEBUG', cat_attr, tuple_1_val, tuple_2_val, common_ancestor_depth_1, common_ancestor_depth_2)
    common_ancestor_depth = common_ancestor_depth_1 if common_ancestor_depth_1 > common_ancestor_depth_2 else common_ancestor_depth_2
    return common_ancestor_depth / CAT_ATTRS[cat_attr]['depth']


def build_attrs_freq(dataset: pandas.DataFrame, total_size: int):
    attrs_freq = {}    
    for attr in QUASI_ATTRIBUTES:
        if attr not in NUMERIC_ATTRS:
            attrs_freq[attr] = {}
            freq_series = dataset[attr].value_counts()
            for index, value in freq_series.items():
                attrs_freq[attr][index] = value / total_size

    return attrs_freq


def cal_tuples_grade(dataset: pandas.DataFrame, attrs_freq: dict, k: int):
    grades = []
    for index, row in dataset.iterrows():
        row_grade = 0
        for attr in QUASI_ATTRIBUTES:
            if attr in NUMERIC_ATTRS:   # If it is a numerical attribute
                row_grade += row[attr] / k
            else:   # If it is a categorical attribute
                row_grade += attrs_freq[attr][row[attr]]
        grades.append(row_grade)

    dataset['grade'] = grades
    return dataset


def cal_tuples_distance(tuple_1: pandas.Series, tuple_2: pandas.Series):
    dist = 0
    for attr in QUASI_ATTRIBUTES:
        if attr in NUMERIC_ATTRS:
            dist += abs(tuple_1[attr] - tuple_2[attr]) / NUMERIC_ATTRS[attr]
        else:
            dist += cal_depth_common_ancestor(attr, tuple_1[attr], tuple_2[attr])

    return dist


def find_cluster_neighbors(anchor_tuple: pandas.Series, dataset: pandas.DataFrame):
    print('Find k-1 neighbors for the tuple {}'.format(anchor_tuple.index))
    distances = []
    for index, row in dataset.iterrows():
        dist = cal_tuples_distance(anchor_tuple, row)
        distances.append(dist)

    dataset['distances'] = distances
    dataset.sort_values(by='distances', inplace=True)
    return dataset


if __name__ == '__main__':
    if len(sys.argv) > 3:
        abs_data_path, oka_abs_output_path, initial_rules_path, k = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
        log_to_file = True
    else:
        k = 10
        abs_data_path, initial_rules_path = 'D:/data_anonymity/dataset/adult-prep.data', 'adult-prep-rules-picked.data'
        oka_abs_output_path = 'D:/data_anonymity/output/out_gccg_k_{}_adult-prep.data'.format(k)
        log_to_file = False
    
    if log_to_file:
        sys.stdout = open("log/gccg_results_k_" + str(k) + ".log", "w")

    R_initial = []
    with open(initial_rules_path, 'rb') as f:
        R_initial = pickle.load(f)

    # for k in [5]:
    #     print('K=', k)
    #     start_time = time.time()

    #     output_file_name = 'out_gccg_k_' + str(k) + '_' + abs_data_path.split('/')[-1].split('.')[0] + '.data'
    #     run_oka_with_adult_ds(abs_data_path, k, output_file_name, log_to_file)

    #     dataset = pandas.read_csv(oka_abs_output_path, names=RETAINED_DATA_COLUMNS, index_col=False, skipinitialspace=True)
    #     GROUPS = dataset.groupby(QUASI_ATTRIBUTES)
    #     total_time = time.time() - start_time

    #     eval_results(R_initial, GROUPS, oka_abs_output_path, total_time, other_algo=True, k=k)

    # sys.stdout.close()

    dataset = pandas.read_csv(abs_data_path, names=RETAINED_DATA_COLUMNS, index_col=False, skipinitialspace=True)
    total_size = dataset.shape[0]
    attrs_freq = build_attrs_freq(dataset, total_size)
    num_of_groups = int(total_size / k - 1)
    print('At initial: Number of groups =', num_of_groups)
    # Grading
    dataset = cal_tuples_grade(dataset, attrs_freq, k)    
    # Centering
    dataset.sort_values(by='grade', ascending=False, inplace=True)
    clusters = []
    first_tuple = dataset.iloc[0]
    find_cluster_neighbors(first_tuple, dataset)
    # for i in range(num_of_groups):

    print(dataset.head(10))
    # print(cal_tuples_distance(dataset.loc[0], dataset.loc[1]))

