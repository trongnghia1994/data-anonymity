"""
read adult data
"""

#!/usr/bin/env python
# coding=utf-8

# Read data and read tree fuctions for INFORMS data
# attributes ['age', 'workcalss', 'final_weight', 'education', 'education_num', 'matrital_status', 'occupation',
# 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'class']
# QID ['age', 'workcalss', 'education', 'matrital_status', 'race', 'sex', 'native_country']
# SA ['occopation']
from oka_py3.models.gentree import GenTree
from oka_py3.models.numrange import NumRange
from oka_py3.utils.utility import cmp_str
from functools import cmp_to_key
import pickle

import pdb

# ATT_NAMES = ['age', 'workclass', 'final_weight', 'education',
#              'education_num', 'marital_status', 'occupation', 'relationship',
#              'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week',
#              'native_country', 'class']
ATT_NAMES = RETAINED_DATA_COLUMNS = ['age', 'sex', 'marital-status', 'native-country',
                'race', 'education', 'hours-per-week', 'capital-gain', 'workclass']
# 8 attributes are chose as QI attributes
# age and education levels are treated as numeric attributes
# only matrial_status and workclass has well defined generalization hierarchies.
# other categorical attributes only have 2-level generalization hierarchies.
QI_INDEX = [0, 1, 2, 3, 4, 5]
IS_CAT = [False, True, True, True, True, True]
SA_INDEX = [6, 7, 8]
DATASET_PATH = 'dataset/adult-prep.data'

__DEBUG = False


def read_data(ds_path=DATASET_PATH):
    """
    read microda for *.txt and return read data
    """
    QI_num = len(QI_INDEX)
    data = []
    numeric_dict = []
    for i in range(QI_num):
        numeric_dict.append(dict())
    # oder categorical attributes in intuitive order
    # here, we use the appear number
    data_file = open(ds_path, 'rU')
    count = 0
    for line in data_file:
        line = line.strip()
        # remove empty and incomplete lines
        # only 30162 records will be kept
        if len(line) == 0 or '?' in line:
            continue
        # remove double spaces
        line = line.replace(' ', '')
        temp = line.split(',')
        ltemp = []
        for i in range(QI_num):
            index = QI_INDEX[i]
            if IS_CAT[i] is False:
                try:
                    numeric_dict[i][temp[index]] += 1
                except:
                    numeric_dict[i][temp[index]] = 1
            ltemp.append(temp[index])
        for i in range(len(SA_INDEX)):
            ltemp.append(temp[SA_INDEX[i]])

        data.append(ltemp)
        count += 1
    # pickle numeric attributes and get NumRange
    for i in range(QI_num):
        if IS_CAT[i] is False:
            static_file = open('oka_py3/data/adult_' + ATT_NAMES[QI_INDEX[i]] + '_static.pickle', 'wb')
            sort_value = list(numeric_dict[i].keys())
            sort_value.sort(key=cmp_to_key(cmp_str))
            pickle.dump((numeric_dict[i], sort_value), static_file)
            static_file.close()
    return data


def read_tree():
    """read tree from data/tree_*.txt, store them in att_tree
    """
    att_names = []
    att_trees = []
    for t in QI_INDEX:
        att_names.append(ATT_NAMES[t])
    for i in range(len(att_names)):
        if IS_CAT[i]:
            att_trees.append(read_tree_file(att_names[i]))
        else:
            att_trees.append(read_pickle_file(att_names[i]))
    return att_trees


def read_pickle_file(att_name):
    """
    read pickle file for numeric attributes
    return numrange object
    """
    try:
        static_file = open('oka_py3/data/adult_' + att_name + '_static.pickle', 'rb')
        (numeric_dict, sort_value) = pickle.load(static_file)
    except:
        print("Pickle file not exists!!")
    static_file.close()
    result = NumRange(sort_value, numeric_dict)
    return result


def read_tree_file(treename):
    """read tree data from treename
    """
    leaf_to_path = {}
    att_tree = {}
    prefix = 'oka_py3/data/adult_'
    postfix = ".txt"
    treefile = open(prefix + treename + postfix, 'rU')
    att_tree['*'] = GenTree('*')    
    for line in treefile:
        # delete \n
        if len(line) <= 1:
            break
        line = line.strip()
        temp = line.split(';')
        # copy temp
        temp.reverse()
        for i, t in enumerate(temp):
            isleaf = False
            if i == len(temp) - 1:
                isleaf = True
            # try and except is more efficient than 'in'
            try:
                att_tree[t]
            except KeyError:
                att_tree[t] = GenTree(t, att_tree[temp[i - 1]], isleaf)    
    treefile.close()
    return att_tree
