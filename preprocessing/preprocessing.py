# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 21:42:55 2019

@author: XUA
"""

from collections import defaultdict

def dict_merge_key(*args):
    """
    Merges many dictionaries by their keys.
    
    Parameters
    ----------
    *args:
        d1 : Dictionary data type
            First dictionary to merge
        d2 : Dictionary data type
            Second dictionary to merge
        d3 : Dictionary data type
        so on:

    Returns
    -------
    dd : Dictionary data type
        Dictionary where d1 and d2 are merged by their keys
    
    Example
    -------
    >>> d1 = {'a':1, 'b':3}
    >>> d2 = {'a':4}
    >>> d3 = {'a':6}
    >>> dict_merge_key(d1, d2, d3)
    ... # doctest: +ELLIPSIS
    {'a': [1, 4, 6], 'b': [3]}
    """
    dd = defaultdict(list)
    for d in args: # you can list as many input dicts as you want here
        for key, value in d.items():
            dd[key].append(value)
    return(dict(dd))