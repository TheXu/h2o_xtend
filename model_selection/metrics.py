# -*- coding: utf-8 -*-
"""
h2o-xtend Machine Learning Library Extensions
Created on 2019

@author: Alex Xu <ayx2@case.edu>
"""
import pandas as pd

def RankingMetricScorer(model, ranking_metric_function, test_data,
                          xval=False):
    """
    Use Ranking Metric function (as in sklearn) for h2o models

    model : h2o.model.model_base.ModelBase

    ranking_metric_function : function

    test_data : H2OFrame

    xval : bool
    """
    print('H2O Algorithm Used: ')
    print('%s %s' %(model.algo, model.type))
    # Specify score column
    if model.type == 'classifier':
        # Only for binary classifier right now
        score_col = 'p1'
    elif model.type == 'regressor':
        score_col = 'predict'
    elif model.type == 'unsupervised':
        score = None
    if xval==False:
        print('\nNot using cross validation metrics')
        # Get scores
        score = model.predict(test_data)[score_col].as_data_frame()
    elif xval==True:
        print('\nUsing Training Data to validate cross validation predictions')
        # Get scores
        score = model.cross_validation_holdout_predictions()[score_col].\
            as_data_frame()
    # Calculate average precision with graph
    # Input parameters on y_true and y_score by position
    ranking_metric =\
    ranking_metric_function(
            test_data[model._model_json["response_column_name"]].\
            as_data_frame().dropna(),
            score
            )
    return(ranking_metric)


def RankingMetricScorerTrainValXval(model, ranking_metric_function,
                                    training_frame, validation_frame=None):
    """
    Ranking metrics for Training, Cross-Validation and Validation data
    """
    train_metric = RankingMetricScorer(model, ranking_metric_function,
                                       training_frame, xval=False)
    xval_metric = RankingMetricScorer(model, ranking_metric_function,
                                      training_frame, xval=True)
    # Put the metrics together into a dictionary
    evaluation_dict = {'train_' + ranking_metric_function.__name__ : train_metric,
                  'xval_' + ranking_metric_function.__name__ : xval_metric}
    if validation_frame != None:
        valid_metric = RankingMetricScorer(model, ranking_metric_function,
                                      validation_frame, xval=False)
        evaluation_dict['valid_' + ranking_metric_function.__name__] =\
            valid_metric
    # Put into a Pandas Dataframe with model_id
    ranking_metric_df = pd.DataFrame(evaluation_dict, index = [model.model_id])
    return(ranking_metric_df)
