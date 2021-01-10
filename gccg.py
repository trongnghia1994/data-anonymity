from common import pick_random_rules, pprint_rule_set, QUASI_ATTRIBUTES, CAT_TREES, RETAINED_DATA_COLUMNS, metrics_cavg, pprint_rule, metrics_cavg_raw, rules_metrics
from ar_mining import cal_supp_conf
from preprocess import preprocess
from Apriori import apriori_gen_rules
from eval import eval_results
from numpy import array_split
from multiprocessing import Pool
from tabulate import tabulate
import pickle, subprocess, time, pandas, sys, time
import itertools, os


NO_PROCESSORS = 4


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

# CAT_COMMON_ANCESTOR_CACHE = {}
def pprint(df):
    print(tabulate(df, headers='keys', tablefmt='psql'))


def intersection(lst1, lst2): 
    lst3 = [value for value in lst1 if value in lst2] 
    return lst3


def find_common_ancestor(cat_attr: str, tuple_values: list):
    # Does not hit cache
    tuple_val_branches = []
    for branch in CAT_ATTRS[cat_attr]['data']:        
        for tuple_val in tuple_values:
            if branch[0] == tuple_val:
                tuple_val_branches.append(branch)

    common_ancestor = tuple_val_branches[0]
    for branch in tuple_val_branches[1:]:
        common_ancestor = intersection(common_ancestor, branch)

    if not common_ancestor:
        raise Exception('Cannot find a common ancestor for {} and {}'.format(tuple_1_val, tuple_2_val))

    return common_ancestor[0]


def cal_depth_common_ancestor(cat_attr, tuple_1_val, tuple_2_val):
    # Does not hit cache
    for branch in CAT_ATTRS[cat_attr]['data']:
        if branch[0] == tuple_1_val:
            tuple_1_val_branch = branch
        if branch[0] == tuple_2_val:
            tuple_2_val_branch = branch

    common_ancestor = intersection(tuple_1_val_branch, tuple_2_val_branch)
    if not common_ancestor:
        raise Exception('Cannot find a common ancestor for {} and {}'.format(tuple_1_val, tuple_2_val))

    common_ancestor = common_ancestor[0]
    common_ancestor_depth_1 = tuple_1_val_branch.index(common_ancestor)
    common_ancestor_depth_2 = tuple_2_val_branch.index(common_ancestor)
    # print('DEBUG', cat_attr, tuple_1_val, tuple_2_val, common_ancestor_depth_1, common_ancestor_depth_2)
    common_ancestor_depth = common_ancestor_depth_1 if common_ancestor_depth_1 > common_ancestor_depth_2 else common_ancestor_depth_2
    result = common_ancestor_depth / CAT_ATTRS[cat_attr]['depth']

    return result


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


def cal_tuples_distance(tuple_1: pandas.Series, tuple_2: pandas.Series, CAT_COMMON_ANCESTOR_CACHE: dict):
    dist = 0
    for attr in QUASI_ATTRIBUTES:
        if attr in NUMERIC_ATTRS:
            dist += abs(tuple_1[attr] - tuple_2[attr]) / NUMERIC_ATTRS[attr]
        else:
            # dist += cal_depth_common_ancestor(attr, tuple_1[attr], tuple_2[attr])
            sorted_key = sorted([tuple_1[attr], tuple_2[attr]])
            key = '{}_{}'.format(attr, '_'.join(sorted_key))
            if key not in CAT_COMMON_ANCESTOR_CACHE:
                print('TOO FAILED', key)
            dist += CAT_COMMON_ANCESTOR_CACHE[key]

    return dist


def cal_distance_worker(args):
    anchor_tuple, ds_part, CAT_COMMON_ANCESTOR_CACHE = args
    distances = []
    for index, row in ds_part.iterrows():
        dist = cal_tuples_distance(anchor_tuple, row, CAT_COMMON_ANCESTOR_CACHE)
        distances.append(dist)

    return distances


def find_cluster_neighbors(anchor_tuple: pandas.Series, dataset: pandas.DataFrame, k: int, CAT_COMMON_ANCESTOR_CACHE: dict):
    # print('Find k-1 neighbors for the tuple {}'.format(anchor_tuple.values))
    distances = []
    # Parallel
    # ds_parts = array_split(dataset, NO_PROCESSORS)    
    # args = [(anchor_tuple, ds_part, CAT_COMMON_ANCESTOR_CACHE) for ds_part in ds_parts]
    # with Pool(processes=NO_PROCESSORS) as pool:
    #     distances = pool.map(cal_distance_worker, args)    
    # distances = list(itertools.chain(*distances))
    # Serial with CUT once finding enough selections
    similar_rows = []
    similar_rows_index = []
    for index, row in dataset.iterrows():        
        dist = cal_tuples_distance(anchor_tuple, row, CAT_COMMON_ANCESTOR_CACHE)
        # Check similar rows to CUT
        if int(dist) == 0:
            similar_rows.append(row)
            similar_rows_index.append(index)
        if len(similar_rows) == k - 1:         
            dataset.drop(similar_rows_index, inplace=True)
            return pandas.DataFrame(similar_rows)

        distances.append(dist)
        
    dataset['distances'] = distances
    dataset.sort_values(by='distances', inplace=True)
    # Get first k tuples
    result = dataset[:k-1]
    dataset.drop(dataset.head(k-1).index, inplace=True)
    # print('Finish 1 iteration in {} seconds'.format(str(time.time() - st)))
    return result


def gccg(dataset: pandas.DataFrame, k: int, gccg_output_path: str):
    total_size = dataset.shape[0]
    attrs_freq = build_attrs_freq(dataset, total_size)

    CAT_COMMON_ANCESTOR_CACHE = {}
    for attr in CAT_ATTRS:
        all_attr_values = attrs_freq[attr].keys()
        for out_attr_val in all_attr_values:
            for in_attr_val in all_attr_values:
                sorted_key = sorted([out_attr_val, in_attr_val])
                key = '{}_{}'.format(attr, '_'.join(sorted_key))
                if key not in CAT_COMMON_ANCESTOR_CACHE:
                    CAT_COMMON_ANCESTOR_CACHE[key] = cal_depth_common_ancestor(attr, out_attr_val, in_attr_val)
    
    # print('CAT_COMMON_ANCESTOR_CACHE', CAT_COMMON_ANCESTOR_CACHE)

    num_of_groups = int(total_size / k)
    print('At initial: Number of groups =', num_of_groups)
    # 1. Grading
    dataset = cal_tuples_grade(dataset, attrs_freq, k)    
    # 2. Centering
    dataset.sort_values(by='grade', ascending=False, inplace=True)
    clusters = []
    st0 = time.time()
    for i in range(num_of_groups):
        st = time.time()
        picked_centroid_tuple = dataset.iloc[0]   
        dataset = dataset[1:]
        # Find k-1 neighbors for picked_centroid_tuple then remove them from the dataset
        result = find_cluster_neighbors(picked_centroid_tuple, dataset, k, CAT_COMMON_ANCESTOR_CACHE)
        result = result.append(picked_centroid_tuple)
        clusters.append(result)

        # if i % 100 == 0:
        #     print('Remaining dataset length: {}'.format(dataset.shape[0]))
        #     print('Time elapsed: {} seconds'.format(time.time() - st))

    # print('Total time: {} seconds'.format(time.time() - st0))
    # 3. Generalization and generate the result
    result_df = pandas.DataFrame()
    for cluster in clusters:
        quasi_values = []
        for attr in QUASI_ATTRIBUTES:
            if attr in NUMERIC_ATTRS:
                quasi_val = '{}-{}'.format(cluster[attr].min(), cluster[attr].max())
                quasi_values.append(quasi_val)
            else:
                all_cat_values = cluster[attr].tolist()
                common_ancestor = find_common_ancestor(attr, all_cat_values)
                quasi_values.append(common_ancestor)

        # pprint(cluster)
        # print(quasi_values)
        # Update all records in this cluster with the quasi values
        for _, row in cluster.iterrows():
            # row.update(quasi_values)
            new_row = pandas.Series(quasi_values + row.values.tolist()[6:9])
            result_df = result_df.append(new_row, ignore_index=True)
            # pprint(result_df.head(5))

    # Handle the remaining records
    for _, row in dataset.iterrows():
        quasi_values = result_df.iloc[0].values.tolist()[:len(QUASI_ATTRIBUTES)]
        new_row = pandas.Series(quasi_values + row.values.tolist()[6:9])
        result_df = result_df.append(new_row, ignore_index=True)
    
    result_df.to_csv(gccg_output_path, index=False, header=False)


if __name__ == '__main__':
    if len(sys.argv) > 3:
        abs_data_path, gccg_output_path, initial_rules_path, k = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
        log_to_file = True
    else:
        k = 10
        abs_data_path, initial_rules_path = 'D:/data_anonymity/dataset/adult-prep.data', 'adult-prep-rules-picked.data'
        gccg_output_path = 'D:/data_anonymity/output/out_gccg_k_{}_adult-prep.data'.format(k)
        log_to_file = False
    
    if log_to_file:
        sys.stdout = open("log/gccg_results_k_" + str(k) + ".log", "w")

    R_initial = []
    with open(initial_rules_path, 'rb') as f:
        R_initial = pickle.load(f)

    for k in [25, 30]:
        print('K=', k)
        start_time = time.time()

        # output_file_name = 'out_gccg_k_' + str(k) + '_' + abs_data_path.split('/')[-1].split('.')[0] + '.data'
        gccg_output_path = 'D:/data_anonymity/output/out_gccg_k_{}_adult-prep.data'.format(k)
        dataset = pandas.read_csv(abs_data_path, names=RETAINED_DATA_COLUMNS, index_col=False, skipinitialspace=True)
        gccg(dataset, k, gccg_output_path)
        total_time = time.time() - start_time

        dataset = pandas.read_csv(gccg_output_path, names=RETAINED_DATA_COLUMNS, index_col=False, skipinitialspace=True)        
        GROUPS = dataset.groupby(QUASI_ATTRIBUTES)      

        eval_results(R_initial, GROUPS, gccg_output_path, total_time, other_algo=True, k=k)

    sys.stdout.close()
