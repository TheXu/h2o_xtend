# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 13:08:25 2019

@author: XUA
"""
from h2o.grid.grid_search import H2OGridSearch

from ..model_selection import GridCVResults
from ..model_selection import GridCVResultsWithMaxMetric
from ..model_selection import GridCVResultsRankingMetric

class hyper_opt:
    """
    Flexible Hyperparameter Optimizer for h2o-py
    """
    def __init__(self, y_name, models_list, params_list, col_list,
                        train, valid=None):
        self.y_name = y_name
        self.models_list = models_list
        self.params_list = params_list
        self.col_list = col_list
        # Datasets and Evaluation Parameters
        self.train = train
        self.valid = valid
        # Define Grid Search
        self.grid_search_list = None


    def fit(self, nfolds=None, fold_column=None):
        """
        Flexible AutoML Wrapper on H2o, where one could specify hyperparameters to optimize over.
        Errored out algorithms will be skipped, and their grid searches will not be returned.
        
        Parameters
        ----------
        train : h2oframe
            Training data set
            
        y_name : str
            Target, Dependent, or Predicted Variable
            
        models_list : list
            list of h2o model objects (trained or untrained; this will not retrain over the object)
        
        params_list : list
            list of dictionary objects for hyperparameters to train for each respective model in models_list
        
        col_list : list
            list of lists of str or numeric datatypes, representing predictory or independent variables to use
            for each respective model in models_list
        
        Returns
        -------
        grid_search_trainned_list : list
            list of h2o grid search objects (h2o.grid.grid_search.H2OGridSearch) that are trained
        """
        # Zipped List of Models, Hyperparameters to Tune, and Column names
        mpc_list = list(zip(self.models_list, self.params_list, self.col_list))

        # Run Grid Seearch on Model with their respective hyperparameters and column names, using their zipped list
        def run_grid_search(model_list):
            """
            Parameters
            ----------
            model_list : tuple
                Stores h2o model object, dictionary of hyperparameters, and list of column names
            """
            # Print Model Used
            print('\n--------------------------------------------------')
            print('H2O Algorithm Used: ')
            print(model_list[0].algo)
            print(model_list[0].params)
            print('\n')
            # Dataset columns and Dataset used for training models
            colnames_X = model_list[2]
            print('Dataset Columns used are: ')
            print(colnames_X)
            print('\n')
            # Hyperparameters
            params = model_list[1]
            print('Hyperparameters Grid Searched Over: ')
            print(params)
            print('\n')
            # Execute Grid Search
            search_criteria = {'strategy': 'Cartesian'}
            grid_search = H2OGridSearch(model=model_list[0],
                                         hyper_params=params,
                                         search_criteria=search_criteria)
            # If Errors out, then return 
            try:
                grid_search.train(x=colnames_X, y=self.y_name,
                                  training_frame=self.train,
                                  validation_frame=self.valid,
                                  nfolds=nfolds, fold_column=fold_column,
                                  keep_cross_validation_predictions = True)
                print('Grid Search Results are:')
                print(grid_search)
                print('\n')
                return(grid_search)
            except Exception as e:
                print('%s Errored Out! Skipping this algorithm!' %model_list[0].algo)
                print(e)
                print('\n')

        # Fit all models for all datasets, Return their Grid Search
        grid_search_trained_list =\
            list(map(run_grid_search, mpc_list))
        # Get rid of any NoneType's (from errored out models)
        grid_search_trained_list =\
            [x for x in grid_search_trained_list if x is not None]
        # Store Grid Search in the class
        self.grid_search_list = grid_search_trained_list


    def AllGridCVResults(self, metric, pprint=False):
        """
        Return Grid Search Results Datasets in a list on specified metric
        for all grid searches for every model specified
        
        Parameters
        ----------

        Returns
        -------
        """
        if self.grid_search_list==None:
            print('Grid Search has not been performed yet\n')
        else:
            if self.valid==None:
                valid = False
            else:
                valid = True
            # Define metric function
            metric_function = lambda g: GridCVResults(g, metric, valid, pprint)
            grid_search_results_list = list(map(metric_function,
                     self.grid_search_list))
        return(grid_search_results_list)


    def AllGridCVResultsWithMaxMetric(self, maxmetric_name=None,
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
        grid_search_results_list : list
            list of Datasets of performance metrics for train, xval, valid
        """
        if self.grid_search_list==None:
            print('Grid Search has not been performed yet\n')
        else:
            if self.valid == None:
                valid = False
            else:
                valid = True
            # Define metric function
            metric_function = lambda g: GridCVResultsWithMaxMetric(g,
                                        maxmetric_name, valid,
                                        pprint)
            grid_search_results_list = list(map(metric_function,
                     self.grid_search_list))
        return(grid_search_results_list)


    def AllGridCVResultsRankingMetric(self, ranking_metric_function,
                                      pprint=False):
        """
        Create Results Data Frame on performance of chosen ranking metric

        Parameters
        ----------
        grid_search : h2o.grid.grid_search.H2OGridSearch

        ranking_metric_function : function
            Ranking metric function where the first two parameters are True
            Target Values, and Predicted Score Values

        Returns
        -------
        grid_search_results_list : list
            list of Datasets of performance metrics for train, xval, valid
        """
        if self.grid_search_list==None:
            print('Grid Search has not been performed yet\n')
        else:
            # Define metric function
            metric_function = lambda model: GridCVResultsRankingMetric(model,
                                                ranking_metric_function,
                                                self.train, self.valid,
                                                pprint)
            grid_search_results_list = list(map(metric_function,
                     self.grid_search_list))
        return(grid_search_results_list)
