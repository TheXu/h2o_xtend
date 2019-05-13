# -*- coding: utf-8 -*-
"""
Created on Fri May  3 10:44:42 2019

@author: XUA
"""
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.utils.fixes import signature

from ..model_selection import RankingMetricScorer

def _precision_recall_curve_(y_true, prob):
    """
    Create precision-recall curve for a 2-Class Classifier and return the Average Precision Score
    
    Parameters
    ----------
    y_true : array-like
        True Labels of Dependent Variable
        
    prob : array-like
        Probability of Class==1
    
    Returns
    -------
    aps : float64
        Average Precision Score between 0 and 1
    """
    precision, recall, thresholds = precision_recall_curve(y_true, prob)
    average_precision = average_precision_score(y_true, prob)
    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
    plt.figure()
    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
              average_precision))
    # Return Average Precision Score
    return(average_precision)


def AveragePrecisionScore(model, test_data, xval=False):
    """
    model : h2o.model.model_base.ModelBase

    test_data : H2OFrame

    xval : bool
    """
    average_precision =\
        RankingMetricScorer(model, _precision_recall_curve_, test_data, xval)
    return(average_precision)
