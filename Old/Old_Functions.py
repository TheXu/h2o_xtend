# -*- coding: utf-8 -*-

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