# --------------------------------------------------------------------------------------------
# Created by Miryam Elizabeth Villa-Pérez
# 
# Source code based on the article:
#
# M. E. Villa-Pérez and L. A. Trejo, "m-OCKRA: An Efficient One-Class Classifier for Personal
# Risk Detection, Based on Weighted Selection of Attributes", IEEE Access 8, pp. 41749-41763,
# 2020, [online] Available: https://doi.org/10.1109/ACCESS.2020.2976947
#
#
#
# Weighted Selection of Attributes based on the metrics for OCC feature selection developed by:
#
# L.H.N. Lorena, A.C.P.L.F. Carvalho and A.C. Lorena, "Filter Feature Selection for One-Class
# Classification", J Intell Robot Syst 80, pp. 227–243, 2015, [online] Available:
# https://doi.org/10.1007/s10846-014-0101-2
#
#
# Core algorithm RandomMiner based on the algorithm developed by:
#
# J. B. Camiña,  M. A. Medina-Pérez,  R. Monroy, O. Loyola-González, L. A. Pereyra Villanueva,
# L. C. González Gurrola, "Bagging-randomminer:  A  one-class  classifier  for  file  access-based
# masquerade detection", Machine Vision and Applications 30(5), pp. 959–974, 2019,
# [online] Available: https://doi.org/10.1007/s00138-018-0957-4
#
# --------------------------------------------------------------------------------------------


import pandas as pd
import numpy as np
import random

from metrics import InformationScore, InterquartileRange, PearsonCorrelation, IntraClassDistance
from methods import getMean, getMajority, getBorda

from scipy.stats import rankdata

from sklearn.base import BaseEstimator
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import pairwise_distances
from sklearn.utils.validation import check_X_y
from sklearn.utils import resample


class m_OCKRA(BaseEstimator):
    
    def __init__(self, classifier_count=50, bootstrap_sample_percent=0.4, mros_percent=0.4, method='Majority',
                 distance_metric='chebyshev', use_bootstrap_sample_count=True, user_threshold=95):
        
        self.classifier_count = classifier_count
        self.bootstrap_sample_percent = bootstrap_sample_percent
        self.mros_percent = mros_percent
        self.method = method
        self.distance_metric = distance_metric
        self.use_bootstrap_sample_count = use_bootstrap_sample_count
        self.user_threshold = user_threshold

    
    def score_samples(self, X):
        
        X_test = pd.DataFrame(X)
        X_test = pd.DataFrame(self._scaler.transform(X_test[X_test.columns]), index=X_test.index, columns=X_test.columns)
        
        similarity = np.average([np.exp(-0.5 * np.power(np.amin(pairwise_distances(X_test[self._features_consider[i]], self._mros[i], metric=self.distance_metric), axis=1) / self._dist_threshold[i], 2)) for i in range(len(self._mros))], axis=0)
        
        return similarity


    def predict(self, X):

        if (len(X.shape) < 2):
            raise ValueError('Reshape your data')

        if (X.shape[1] != self.n_features_):
            raise ValueError('Reshape your data')

        if not self._is_threshold_Computed:
            
            x_pred_classif = self.score_samples(X)            
            x_pred_classif.sort()
            self._inner_threshold = x_pred_classif[(100 - self.user_threshold) * len(x_pred_classif) // 100]
            self._is_threshold_Computed = True
        
        y_pred_classif = self.score_samples(X)
        
        return [-1 if s <= self._inner_threshold else 1 for s in y_pred_classif]
    
    
    def weightedFeatureSelection(self, X):
                
        scores_list = []

        scores_list.append(InterquartileRange(X))
        scores_list.append(PearsonCorrelation(X, self.n_features_))
        scores_list.append(IntraClassDistance(X, self.n_objects_, self.n_features_))
        scores_list.append(InformationScore(X, self.n_objects_, self.n_features_))
        
        
        # All values are given a distinct rank, corresponding to the order that the values occur in the array
        ranks = [rankdata(score, method='ordinal') for score in scores_list]
        
        r = []
        if self.method == 'Mean':
            r = getMean(ranks)
        
        elif self.method == 'Majority':
            r = getMajority(ranks)
        
        elif self.method == 'Borda':
            r = getBorda(ranks)
        
        else:
            raise ValueError('Aggregation method does not exist!')
        

        values_sort = np.array((max(r) + 1) - r)
        
        lst = sum([[x - 1] * x for x in values_sort], [])
               
        
        return [np.unique(np.random.choice(lst, self.n_features_)) for x in range(self.classifier_count)]
    

    def fit(self, X, y):
        
        # Check that X and y have correct shape
        X_train, y_train = check_X_y(X, y)
        
        self._is_threshold_Computed = False
        
        # Total of features in dataset
        self.n_objects_, self.n_features_ = X_train.shape
        
        if self.n_features_ < 1:
            raise ValueError('Unable to instantiate the train dataset - Empty vector')
        
        self._scaler = MinMaxScaler()
        X_train = pd.DataFrame(X_train)
        X_train = pd.DataFrame(self._scaler.fit_transform(X_train[X_train.columns]), index=X_train.index, columns=X_train.columns)


        # Random features
        self._features_consider = self.weightedFeatureSelection(X_train)
        
        # Save centers clustering and threshold distance
        self._mros = []
        self._dist_threshold = np.empty(self.classifier_count)
        
        sampleSizeBootstrap = int(self.bootstrap_sample_percent * len(X_train)) if (self.use_bootstrap_sample_count) else int(0.01 * len(X_train));
        sampleSizeMros = int(self.bootstrap_sample_percent * sampleSizeBootstrap) if (self.use_bootstrap_sample_count) else int(0.01 * sampleSizeBootstrap);


        for i in range(self.classifier_count):
            
            projected_dataset = X_train[self._features_consider[i]]
            
            # RandomMiner
            # 1. Bootstrap (Random sample with replacement)
            # 2. MROs (Random sample without replacement)
            bootstrap = resample(projected_dataset, n_samples=sampleSizeBootstrap)
            mros = resample(bootstrap, n_samples=sampleSizeMros, replace=False)
            self._mros.append(mros.values)
            
            
            # Distance threshold
            self._dist_threshold = np.insert(self._dist_threshold, i, 1 - np.sum(self._features_consider[i] / self.n_features_))
        
        return self
