# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 14:11:35 2019

@author: XUA
"""

from random import shuffle
import numpy as np
import pandas as pd

def GroupKFoldAssign(df, group, n_folds=3):
    """
    Create fold numbers for each unique group of a dataset.
    Return unique groups with associated fold number. Once merged could be
    passed into "fold_column" of train functions

    Parameters
    ----------
    df : pandas.dataframe

    group : str
        Column of "df" to perform Group Cross Validation on

    n_folds : int, optional
        Number of folds

    Returns
    -------
    fold_df : pandas.dataframe
        Dataset where each unique group has a fold number assigned to it

    Examples
    --------
    >>> from h2o_xtend.model_selection import GroupKFoldAssign
    >>> import h2o
    Not complete...
    """
    # Create group list
    uniq = df[group].unique().tolist()
    no_of_group = len(uniq)
    # Randomly shuffle our index, inplace
    shuffle(uniq)
    # Floor of the Number of Groups divided by Number of Folds
    groupNo_div_n = int(np.floor(no_of_group/n_folds))
    remainder = no_of_group % n_folds
    # Create fold list based on number of folds
    fold = list(range(n_folds)) * groupNo_div_n + [i for i in range(remainder)]
    # Return data frame of Unique Groups with their Fold Numbers
    fold_df = pd.DataFrame(
            {group:uniq, 'GroupKFoldCol':fold},
                           index=list(range(no_of_group))
                           )
    return(fold_df)