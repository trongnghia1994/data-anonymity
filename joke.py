import pickle, pandas, time
from common import DATA_FILE_PATH, RETAINED_DATA_COLUMNS, QUASI_ATTRIBUTES, CAT_TREES, construct_r_affected_by_a_migration, build_groups, construct_r_care, group_length

# # Read list of rules from binary pickled file
# # with open('initial_rules.data', 'rb') as f:
# #     data = pickle.load(f)
# #     print(data)

# df = pandas.read_csv(DATA_FILE_PATH, names=RETAINED_DATA_COLUMNS,
#                     index_col=False, skipinitialspace=True)
# a = df['education'].unique()
# for i in sorted(a):
#     print("['{}'-'*'],".format(i))

# df = pandas.read_csv('D:/data_anonymity/dataset/adult-prep.data', 
#                     names=RETAINED_DATA_COLUMNS,
#                     index_col=False,
#                     skipinitialspace=True)

# count = df.groupby(QUASI_ATTRIBUTES).size()
# count.to_csv('joke.csv', header=False)

# import pickle

# f_path = "D:/data_anonymity/Clustering_based_K_Anon/data/adult_age_static.pickle"
# with open(f_path, 'rb') as f:
#     print(f)
#     data = pickle.load(f)

# print(data)

# from crowds.kanonymity import ola
# from crowds.kanonymity.generalizations import GenRule
# from crowds.kanonymity.information_loss import dm_star_loss

# from common import DATA_FILE_PATH, RETAINED_DATA_COLUMNS

# Read list of rules from binary pickled file
# with open('initial_rules.data', 'rb') as f:
#     data = pickle.load(f)
#     print(data)

# def loss_fn(node):
#     return 0.0


# def first_gen(value):
#     return '*'

# def second_gen(value):
#     return '*'


# new_rule = GenRule([first_gen, second_gen])
# ruleset = {'marital-status': new_rule}

# def generalize_age(value):
#     min = int(value / 10)
#     max = min + 10
#     return '{}-{}'.format(min, max)


# def generalize_workclass(value):
#     if value in ['Private']:
#         return '*'
#     if value in ['Self-emp-not-inc', 'Self-emp-inc']:
#         return 'Self-employ'
#     if value in ['Federal-gov', 'Local-gov', 'State-gov']:
#         return 'gov'
#     if value in ['Without-pay', 'Never-worked']:
#         return 'not-work'

# def generalize_marital_status(value):
#     if value in ['Never-married']:
#         return '*'
#     if value in ['Married-civ-spouse', 'Married-AF-spouse']:
#         return 'Married'
#     if value in ['Divorced', 'Separated']:
#         return 'leave'
#     if value in ['Widowed', 'Married-spouse-absent']:
#         return 'alone'


# def star_generalize(value):    
#     return '*'


# generalization_rules = {
#     'age': GenRule([generalize_age, star_generalize]), # 3-levels generalization
#     'sex': GenRule([star_generalize]), # 2-level generalization
#     'marital-status': GenRule([generalize_marital_status, star_generalize]),
#     'native-country': GenRule([star_generalize]),
#     'race': GenRule([star_generalize]),
#     'education': GenRule([star_generalize]),    
# }

# df = pandas.read_csv(DATA_FILE_PATH, names=RETAINED_DATA_COLUMNS,
#                     index_col=False, skipinitialspace=True)

# anonymous_df = ola.anonymize(df, k=10, info_loss=dm_star_loss, generalization_rules=generalization_rules)
# out_file_path = 'output/out_ola_k_{}_adult-prep.data'.format(k)
# anonymous_df[0].to_csv(out_file_path, index=False, header=False)          


# R_AFFECTED = {}
# data_file_path = 'dataset/adult-prep.data'
# initial_rules_path = 'adult-prep-rules-picked.data'
# k = 10

# def write_to_file(data, no_tuples, k):
#     file_name = 'r_affected/k_{}_no_groups_{}.data'.format(k, no_tuples) 
#     with open(file_name, 'wb') as f:
#         pickle.dump(data, f)


# from multiprocessing import Pool
# import os, pymongo

# k = 10
# A dataset reaches k-anonymity if total risks of all groups equals to 0
# A Member Migration operation g(i)-T-g(j) is valuable when the risk of data is decreased after performing that Member Migration operation.


# myclient = pymongo.MongoClient("mongodb://localhost:27017/")
# mydb = myclient["data_anonymity"]
# mycol = mydb["rules_affected"]


def worker(data):
    sub_groups, groups, R_care = data
    r = {}
    i = 0
    for gr_out in sub_groups:
        for gr_in in groups:
            # print(os.getpid(), gr_in.index, gr_out.index)
            if gr_out.index != gr_in.index:
                key = '{}_{}'.format(gr_out.index, gr_in.index)
                tuples_consider_to_move = gr_out.origin_tuples[0]
                # r[key] = construct_r_affected_by_a_migration(R_care, [tuples_consider_to_move], gr_in)    
                r_affected = construct_r_affected_by_a_migration(R_care, [tuples_consider_to_move], gr_in)    
                mydict = { "_id": key, "r": pickle.dumps(r_affected) }
                mycol.insert_one(mydict)
        i += 1
        if i % 50 == 0:
            print('Process {} completes {} groups'.format(os.getpid(), i))

    for gr_out in groups:
        for gr_in in sub_groups:
            # print(os.getpid(), gr_in.index, gr_out.index)
            if gr_out.index != gr_in.index:
                key = '{}_{}'.format(gr_out.index, gr_in.index)
                tuples_consider_to_move = gr_out.origin_tuples[0]
                # r[key] = construct_r_affected_by_a_migration(R_care, [tuples_consider_to_move], gr_in)    
                r_affected = construct_r_affected_by_a_migration(R_care, [tuples_consider_to_move], gr_in)    
                mydict = { "_id": key, "r": pickle.dumps(r_affected) }
                mycol.insert_one(mydict)
        i += 1
        if i % 50 == 0:
            print('Process {} completes {} groups'.format(os.getpid(), i))

    # file_name = 'r_affected/k_{}_pid_{}.data'.format(k, os.getpid()) 
    # with open(file_name, 'wb') as f:
    #     pickle.dump(r, f)    

# from os import listdir
# from os.path import isfile, join
# R_affected = {}
# def recover_r_affected(path):
#     for f in listdir(path):
#         fp = join(path, f)
#         print('fp', fp)
#         with open(fp,'rb') as f:
#             R_affected.update(pickle.load(f))


if __name__ == '__main__':
    data_file_path = 'dataset/adult-prep.data'
    # initial_rules_path = 'adult-prep-rules-picked.data'    
    D = pandas.read_csv(data_file_path, names=RETAINED_DATA_COLUMNS, index_col=False, skipinitialspace=True)
    dataset_length = D.shape[0]
    print('DATASET LENGTH=', dataset_length)
    # print(D['age'].max(), D['age'].min())
    print(D['workclass'].value_counts())
    # GROUPS, SG, UG, UG_SMALL, UG_BIG = build_groups(D, k=10)
    # groups_col = mydb["groups"]
    # for group in GROUPS:
    #     groups_col.insert_one({"_id": group.index, "data": pickle.dumps(group)})
    # R_initial = []
    # with open(initial_rules_path, 'rb') as f:
    #     R_initial = pickle.load(f)
    # R_care = construct_r_care(R_initial)  # List of cared rules
    # i = 0
    # st = time.time()
    # # Divide GROUPS into 4 parts
    # chunks = [(UG[:2000], GROUPS, R_care), (UG[2000:4000], GROUPS, R_care), (UG[4000:5500], GROUPS, R_care), (UG[5500:], GROUPS, R_care)]
    # # chunks = [(GROUPS[:1000], R_care), (GROUPS[1000:2000], R_care), (GROUPS[2000:3000], R_care), (GROUPS[3000:4000], R_care)
    # # (GROUPS[4000:5000], R_care), (GROUPS[5000:6000], R_care), (GROUPS[6000:7000], R_care), (GROUPS[7000:], R_care)
    # # ]
    # with Pool(processes=4) as pool:
    #     res = pool.map(worker, chunks)

    # recover_r_affected('r_affected')
    # print(R_affected)

    # mydict = { "_id": "John", "address": "Highway 37" }
    # mycol.insert_one(mydict)

    # rules = pickle.loads(mycol.find_one({"_id": "1042_1071"})['r'])
    # print(rules)

    # GROUPS_LENGTH = {}
    # for group in GROUPS:
    #     gr_len = group_length(group)
    #     if gr_len not in GROUPS_LENGTH:
    #         GROUPS_LENGTH[gr_len] = 1
    #     else:
    #         GROUPS_LENGTH[gr_len] += 1

    # print('Lengths', len(UG), len(SG), len(UG_SMALL), len(UG_BIG))
    # print('Group length stats')
    # for key in sorted(GROUPS_LENGTH):
    #     print('Length={}'.format(key), ':', GROUPS_LENGTH[key])
    # from pprint import pprint
    # with open('D:/data_anonymity/oka_py3/data/adult_age_static.pickle', 'rb') as f:
    #     data = pickle.load(f)
    #     pprint(data[0])
