"""
public functions
"""
from functools import cmp_to_key

# !/usr/bin/env python
# coding=utf-8

def cmp(a, b):
    if a < b:
        return -1
    elif a == b:
        return 0
    else:
        return 1


def cmp_str(element1, element2):
    """
    compare number in str format correctly
    """
    try:
        return cmp(int(element1), int(element2))
    except ValueError:
        return cmp(element1, element2)


def qid_to_key(value_list, sep=';'):
    """convert qid list to str key
    value (splited by sep). This fuction is value safe, which means
    value_list will not be changed.
    return str list.
    """
    return sep.join(value_list)


def list_to_str(value_list, cmpfun=cmp, sep=';'):
    """convert sorted str list (sorted by cmpfun) to str
    value (splited by sep). This fuction is value safe, which means
    value_list will not be changed.
    return str list.
    """
    temp = value_list[:]
    temp.sort(key=cmp_to_key(cmpfun))
    return sep.join(temp)


def get_num_list_from_str(stemp):
    """
    if float(stemp) works, return [stemp]
    else return, stemp.split('-')

    """
    try:
        float(stemp)
        return [stemp]
    except ValueError:
        return stemp.split('-')
