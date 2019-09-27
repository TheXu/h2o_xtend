# -*- coding: utf-8 -*-
"""
h2o-xtend Machine Learning Library Extensions
Created on 2019

@author: Alex Xu <ayx2@case.edu>
"""
import pandas as pd
import h2o
from IPython.display import display

from .metrics import RankingMetricScorerTrainValXval

from ..preprocessing import dict_merge_key
from ..plots import std_coef_plot

def GridCVResults(grid_search, metric,
                  valid=False, pprint=False, greater_is_better=True):
    """
    Function that takes an h2o grid search object and extracts training and cross validation
    metrics into a Pandas DataFrame along with grid search id's
    
    Parameters
    ----------
    grid_search : h2o.grid.grid_search.H2OGridSearch
    
    metric : str
        Represents method for H2o Grid Search Object that extracts evaluation metric of choosing
    
    valid : bool, optional
        Whether you want to include validation metrics or not. h2o Grid Search has to be trained on a validation frame
    
    Returns
    -------
    metric_df: pandas.dataframe
        training and cross validation (and validation) metrics into a
        Pandas DataFrame along with grid search id's
    """
    # Obtain method object for model evaluation metric for h2o model
    method_to_call = getattr(grid_search, metric)
    train_dict = method_to_call(train=True)
    xval_dict = method_to_call(xval=True)
    # If we want validation metrics, then call it and merge it!
    if valid == True:
        valid_dict = method_to_call(valid=True)
        # Merge dictionaries by their keys!
        merged_dict = dict_merge_key(train_dict, xval_dict, valid_dict)
        colnames_list = ['train_'+metric, 'xval_'+metric, 'valid_'+metric]
        sort_name = ['valid_'+metric]
    else:
        # Merge dictionaries by their keys!
        merged_dict = dict_merge_key(train_dict, xval_dict)
        colnames_list = ['train_'+metric, 'xval_'+metric]
        sort_name = ['xval_'+metric]
    # Return merged dictionary but as a Pandas DataFrame
    metric_df = pd.DataFrame.from_dict(merged_dict).T
    # Change column names
    metric_df.columns = colnames_list
    metric_df = metric_df.sort_values(sort_name, ascending=not greater_is_better)
    # Display
    if pprint == True:
        display(metric_df)
    return(metric_df)


def ModelIdsResults(model_ids_list, metric,
                    valid=False, pprint=False, greater_is_better=True):
    """
    Function that takes list of model_ids from h2o and extracts training and cross validation
    metrics into a Pandas DataFrame along with model_ids
    
    Parameters
    ----------
    model_ids_list : list
        Contains list of str's, representing model_id's within h2o cluster
        
    metric : str
        Represents method for H2o Grid Search Object that extracts evaluation metric of choosing
    
    valid : bool
        Whether you want to include validation metrics or not. h2o Grid Search has to be trained on a validation frame
    
    Returns
    -------
    metric_df: Pandas DataFrame Object
        training and cross validation (and validation) metrics into a Pandas DataFrame along with grid search id's
    """
    # List of methods for evaluating "metric", associated with each model_id
    methods_list = list(map(lambda x: getattr( h2o.get_model(x), metric ),
                            model_ids_list))
    # List of evaluation metrics for both training set and testing set
    train_list = list(map(lambda x: x(train=True), methods_list))
    xval_list = list(map(lambda x: x(xval=True), methods_list))
    # Make dataframe full of metrics
    metric_df = pd.DataFrame({'train_'+metric:train_list,
                              'xval_'+metric:xval_list},
                             index=model_ids_list)
    if valid:
        valid_list = list(map(lambda x: x(xval=True), methods_list))
        # Create new column with validation metrics
        metric_df['valid_'+metric] = pd.Series(valid_list)
        metric_df.sort_values('valid_'+metric,
                              ascending=not greater_is_better, inplace=True)
    else:
        metric_df.sort_values('xval_'+metric,
                              ascending=not greater_is_better, inplace=True)
    if pprint == True:
        display(metric_df)
    # Return Train/Xval evaluation metrics for each model id
    return(metric_df)


def GridCVResultsWithMaxMetric(grid_search, maxmetric_name=None, valid=False,
                               pprint=False):
    """
    Create Results Data Frame on performance of chosen binary
    classifier metrics

    Different than GridCVResults as there is an option to add a max_metric
    to dataframe

    Parameters
    ----------
    grid_search : h2o.grid.grid_search.H2OGridSearch
    
    maxmetric_name : str
        Metric to use along with 'AUC'. A max metric like max_f1,
        max_precision, etc.
        i.e. If Imbalanced Datset with less positive lables,
        then recommended to use precision-recall related metric
    
    Returns
    -------
    perf3_df : pandas.DataFrame
        Dataset of performance metrics for train, xval, valid
    """
    # Find a model improves distinguishing between classes
    grid_search_perf1 = grid_search.\
        get_grid(sort_by='auc', decreasing=True)
    # Train and Cross Validation AUC's and F1's
    perf3_df = GridCVResults(grid_search_perf1, 'auc', valid=valid)
    if valid == True:
        sort_by_metric = 'valid_' + maxmetric_name
    elif valid == False:
        sort_by_metric = 'xval_' + maxmetric_name

    # If we want to add another metric
    if maxmetric_name is not None:
        # Find a model that improves Precision and Recall Curve
        grid_search_perf2 = grid_search.\
            get_grid(sort_by=maxmetric_name, decreasing=True)
        # Train and Cross Validation AUC's and F1's
        perf2_df =\
            GridCVResults(grid_search_perf2, maxmetric_name, valid=valid).\
            applymap(lambda x: x[0][1]).\
            sort_values([sort_by_metric], ascending=False)
        # Merge
        perf3_df = pd.\
            merge(perf3_df, perf2_df, how='inner',
                  right_index=True, left_index=True)
    # Display
    if pprint == True:
        display(perf3_df)
    return(perf3_df)


def GridCVResultsRankingMetric(grid_search, ranking_metric_function,
                               training_frame, validation_frame=None,
                               pprint=False, greater_is_better=True):
    """
    Function that takes an h2o grid search object and extracts training and cross validation
    metrics into a Pandas DataFrame along with grid search id's

    Parameters
    ----------
    grid_search : h2o.grid.grid_search.H2OGridSearch

    ranking_metric_function : function
        Ranking metric function where the first two parameters are True Target
        Values, and Predicted Score Values

    training_frame : H2OFrame
        Dataset of training dataset

    validation_frame : H2OFrame, optional
        Dataset of validation dataset

    Returns
    -------
    ranking_metric_df: pandas.dataframe
        training and cross validation (and validation) ranking metrics into a
        Pandas DataFrame along with grid search id's
    """
    # Create metric function
    metric_function = lambda model: RankingMetricScorerTrainValXval(model,
                                                          ranking_metric_function,
                                                          training_frame,
                                                          validation_frame)
    # Iterate over grid search models, and put all results into a dataframe
    ranking_metric_df = pd.concat(list(map(
            metric_function, grid_search.models
                                                          )))
    # Sort by Metric defined
    if validation_frame is not None:
        # If we have validation frame, sort by validation metric
        sort_by_metric = 'valid_' + ranking_metric_function.__name__
    elif validation_frame is None:
        # Else sort by xval metrics
        sort_by_metric = 'xval_' + ranking_metric_function.__name__
    ranking_metric_df = ranking_metric_df.sort_values([sort_by_metric],
                                                      ascending=not greater_is_better)
    # Display
    if pprint == True:
        display(ranking_metric_df)
    return(ranking_metric_df)


def get_var_imp(h2o_model, no_top_features=20):
    """
    More general way to get Variable Importances
    """
    print('\n---------------------------------')
    print('Variable Importance for %s: ' % h2o_model.algo)
    if h2o_model.algo == 'glm':
        # Set output name to get feature importances
        # Set reindex structure
        if h2o_model._model_json["output"]["model_category"]=="Multinomial":
            # Reindex using sum of absolute values of all classes' std coefs
            output_name = 'coefficients_table_multinomials_with_class_names'
            reindexByAbsStdCoef =\
                h2o_model.\
                _model_json["output"]["standardized_coefficient_magnitudes"].\
                set_index('names').index
        else:
            output_name = 'coefficients_table'
        # Get Coefficients from h2o glm model and put it in a dataframe
        varimp = h2o_model._model_json['output'][output_name].\
            as_data_frame()
        # Reindex by largest absolute standardized coefficient
        if h2o_model._model_json["output"]["model_category"]!="Multinomial":
            reindexByAbsStdCoef = varimp['standardized_coefficients'].abs().\
                sort_values(inplace=False, ascending=False).\
                index
        # Sort by Absolute Standardized Coefficients
        varimp = varimp.reindex(reindexByAbsStdCoef)
        # Display top features in graph
        if h2o_model._model_json["output"]["model_category"]!="Multinomial":
            # If we aren't dealing with Multinomial Logistic Regression
            # Display one graphs
            display(h2o_model.std_coef_plot(num_of_features=no_top_features))
        elif h2o_model._model_json["output"]["model_category"]=="Multinomial":
            # If dealing with Multinomial Logistic Regression,
            # Display multiple graphs for each class
            for category in h2o_model._model_json["output"]['domains'][len(h2o_model._model_json["output"]['domains'])-1]:
                display(std_coef_plot(h2o_model, category,
                                      num_of_features=no_top_features))
        print('\n')
    else:
        # Get Variables Importances on average split error change
        # (for tree based models)
        varimp = pd.DataFrame(h2o_model.varimp()).\
            rename(index=str,
                   columns=\
                   {0:'variable', 1:'relative_importance',
                    2:'scaled_importance', 3:'percentage_importance'})
        display(h2o_model.varimp_plot(num_of_features=no_top_features))
        print('\n')
    # Return Variable importances
    display(varimp.head(no_top_features))
    print('\n')
    return(varimp)