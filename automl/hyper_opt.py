# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 13:08:25 2019

@author: XUA
"""
import pandas as pd
from h2o.grid.grid_search import H2OGridSearch

from ..preprocessing import dict_merge_key
from ..model_selection import GridCVResults
from ..model_selection import GridCVResultsWithMaxMetric

class hyper_opt:
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
            search_criteria = { 'strategy': "Cartesian" }
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
            list(map(lambda li: run_grid_search(li), mpc_list))
        # Get rid of any NoneType's (from errored out models)
        grid_search_trained_list =\
            [x for x in grid_search_trained_list if x is not None]
        # Store Grid Search in the class
        self.grid_search_list = grid_search_trained_list

    def AllGridCVResults(self, metric):
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
            grid_search_results_list = list(map(lambda g:
                     GridCVResults(g, metric, valid),
                     self.grid_search_list))
        return(grid_search_results_list)

    def AllGridCVResultsWithMaxMetric(self, maxmetric_name=None):
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
        if self.grid_search_list==None:
            print('Grid Search has not been performed yet\n')
        else:
            if self.valid==None:
                valid = False
            else:
                valid = True
            grid_search_results_list = list(map(lambda g:
                     GridCVResultsWithMaxMetric(g, maxmetric_name, valid),
                     self.grid_search_list))
        return(grid_search_results_list)
    
    def get_avg_prec_score(perf_df, test_df, y_label):
        """
        Get Average Precision Score from best model from ordered dataframe of model id's
        
        Parameters
        ----------
        perf_df : pandas.DataFrame
            Dataset of model id's, ordered by performance, with top performing being 'first'
            
        test_df : pandas.DataFrame
            Testing Dataset to get Average Precision Score on
        
        y_label : str
            Target or Dependent Variable Name
        """
        import h2o
        best_model = h2o.get_model( perf_df.index[0] )
        print('H2O Algorithm Used: ')
        print(best_model.algo)
        aps = sklearn.precision_recall_curve(test_df[y_label].as_data_frame().astype(int), best_model.predict(test_df)[2].as_data_frame())
        return(aps)

    def glm_explain(train_hf, X, y_hat_values, distribution, valid_hf, no_top_features=20):
        """
        Global Surrogate Interpretable Model-Agnostic Explanations (GS-IME) using Generalized Linear Models
        """
        # Define metric to sort grid search by
        if distribution=='binomial':
            metric = 'auc'
        else:
            metric = 'mse'
        # Make deep copy of h2o frame
        train_hf = h2o.deep_copy(train_hf, 'GS-IME_hf')
        train_hf['y_hat'] = y_hat_values
        # Assign Lists of modeling strategy
        model_list = [H2OGeneralizedLinearEstimator(family=distribution, solver='L_BFGS', max_iterations=50000, standardize=True
                                              , keep_cross_validation_models=True, keep_cross_validation_fold_assignment=True
                                            , missing_values_handling='mean_imputation')]
        hyper_list = [{'alpha':[0.001, 0.01, 0.1, .9],
                 'lambda': [1e-8,1e-3,0.1]}]
        # Fit One Set of GLM Models and perform grid search
        glm_grid_list = fit_many_models( train=train_hf, y_name='y_hat', models_list=model_list, params_list=hyper_list, col_list=[X], valid=valid_hf )
        # Get top model
        top_model = glm_grid_list[0].get_grid(sort_by=metric, decreasing=True)[0]
        # Get Standardized Coefficients of Global Surrogate Model
        coef = get_var_imp( top_model, no_top_features )
        return(coef)
        
    def many_threshold_metrics(h2o_model, thresh=[0, 0.05, 0.15, 0.25, 0.35, .45, .6, .9]):
        """
        Calculate performance metrics on chosen thresholds
        """
        print('\n----------------------------------------')
        print('H2O Algorithm: %s' %h2o_model.algo)
        display(h2o_model.confusion_matrix( thresholds=thresh ) )
        display(h2o_model.precision( thresholds=thresh ) )
        display(h2o_model.recall( thresholds=thresh ) )
        display(h2o_model.F2( thresholds=thresh ) )
        
    class h2o_predict_proba_wrapper:
    # drf is the h2o distributed random forest object, the column_names is the
    # labels of the X values
        """
        Stole from here: https://marcotcr.github.io/lime/tutorials/Tutorial_H2O_continuous_and_cat.html
        """
        def __init__(self,model,column_names):
                
                self.model = model
                self.column_names = column_names
         
        def predict_proba(self,this_array):        
            # If we have just 1 row of data we need to reshape it
            shape_tuple = np.shape(this_array)        
            if len(shape_tuple) == 1:
                this_array = this_array.reshape(1, -1)
                
            # We convert the numpy array that Lime sends to a pandas dataframe and
            # convert the pandas dataframe to an h2o frame
            self.pandas_df = pd.DataFrame(data = this_array,columns = self.column_names)
            self.h2o_df = h2o.H2OFrame(self.pandas_df)
            
            # Predict with the h2o drf
            self.predictions = self.model.predict(self.h2o_df).as_data_frame()
            # the first column is the class labels, the rest are probabilities for
            # each class
            self.predictions = self.predictions.iloc[:,1:].as_matrix()
            return self.predictions
    
    class average_precision_score:
        """
        Documentation and examples located here:
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html#sklearn.metrics.average_precision_score
            https://github.com/h2oai/h2o-3/blob/master/h2o-docs/src/dev/custom_functions.md
            https://github.com/h2oai/h2o-tutorials/blob/master/tutorials/custom_metric_func/CustomMetricFuncClassification.ipynb
        """
        def map(self, predicted, actual, weight, offset, model):
            from sklearn.metrics import average_precision_score
            y = actual[0]
            p = predicted[2] #[class, p0, p1]
            aps = average_precision_score(y, p)
            return(aps)
        
        def reduce(self, left, right):
            return [left[0] + right[0], left[1] + right[1]]
    
        def metric(self, last):
            return last[0] / last[1]