# -*- coding: utf-8 -*-
"""
Created on Fri May  3 10:44:42 2019

@author: XUA
"""
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, classification_report
from sklearn.calibration import calibration_curve
from sklearn import metrics
from sklearn.metrics import precision_recall_curve, average_precision_score
from funcsigs import signature

def precision_recall_score_curve(y_true, prob):
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


def prob_eval_viz(y_test, y_proba, large=True):
    """
    Set of plots describing various ranking metrics between Actuals and Calibrated/Un-Calibrated Probability Scores

    Parameters
    ----------
    y_test : array-like
        Actuals

    y_proba : array-like
        Predicted Scores

    Returns
    -------
    None
    """
    # create subplot
    if large == True:
        plt.figure(figsize=(9, 14))
    else:
        plt.figure(figsize=(6, 9))
    ax1 = plt.subplot2grid((4,4), (0,0), colspan=2)
    ax2 = plt.subplot2grid((4,4), (0,2), colspan=2)
    ax3 = plt.subplot2grid((4,4), (1, 0), colspan=4)
    ax4 = plt.subplot2grid((4,4), (2, 0), colspan=4)
    
    # Precision-Recall Curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    average_precision = average_precision_score(y_test, y_proba)
    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})
    ax1.step(recall, precision, color='b', alpha=0.2,
             where='post')
    ax1.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
    
    ax1.set_xlabel('Recall')
    ax1.set_ylabel('Precision')
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlim([0.0, 1.0])
    ax1.set_title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
              average_precision))
    # roc curve
    ax2.set_title('ROC Curve', fontsize=14)
    fpr, tpr, _ = metrics.roc_curve(y_test, y_proba)
    area = metrics.auc(fpr, tpr)
    ax2.plot(fpr, tpr, label='(area = %0.2f)' % area, linewidth=3, color='#7C4392')
    ax2.plot([0, 1], [0, 1], 'k--')
    ax2.set_xlabel('False positive rate')
    ax2.set_ylabel('True positive rate')
    ax2.legend(loc='lower right')
    
    # calculate fraction of positives and mean predicted value
    fop, mpv = calibration_curve(y_test, y_proba, n_bins=10)
    
    # calibration plot
    ax3.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    ax3.plot(mpv, fop, "s-", color='#7C4392')
    ax3.set_ylabel('Fraction of positives')
    ax3.set_ylim([-0.05, 1.05])
    ax3.set_title('Calibration Plots (Reliability Curve)', fontsize=14)

    # mean probability by decile plot
    ax4.hist(y_proba, range=(0, 1), bins=10, histtype='stepfilled', lw=2, color='#7C4392')
    ax4.set_xlabel('Mean predicted value')
    ax4.set_ylabel('Frequency')
    
    # tight layout
    plt.tight_layout()
    plt.show()
