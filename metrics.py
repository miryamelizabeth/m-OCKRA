# --------------------------------------------------------------------------------------------
# Metrics developed in the article "Filter Feature Selection for One-Class Classification" by Lorena et al. 2015.
#
# This Python implementation is based on the source code made in Matlab by the authors,
# available at https://github.com/LuizHNLorena/FilterFeatureOneClass
#
# For more information, please read:
# L.H.N. Lorena, A.C.P.L.F. Carvalho and A.C. Lorena, "Filter Feature Selection for One-Class
# Classification", J Intell Robot Syst 80, pp. 227–243, 2015, [online] Available:
# https://doi.org/10.1007/s10846-014-0101-2
# --------------------------------------------------------------------------------------------


import pandas as pd
import numpy as np
import sys

from scipy.stats import iqr
from sklearn.metrics.pairwise import rbf_kernel



# ----------------------------------------------------------------------------
# Features with high values for this index are very correlated to others.
# To favor the maintenance of features that represent more exclusive concepts,
# lower values are preferred.
# ----------------------------------------------------------------------------
def PearsonCorrelation(dataset, columns):

    # Applying Pearson
    ds = pd.DataFrame(dataset)
    correlation = ds.corr(method='pearson')
    dfRHO = correlation.replace(np.nan, 0) # Replace nan with zero values (columns with std = 0)
    RHO = dfRHO.values

    # Evaluating each individual atribute
    pearsonSum = []
    for column in range(columns):
        curRHO = RHO[:, column]
        pearsonSum.append(np.sum(np.abs(curRHO)) - 1)

    pearsonVals = np.array(pearsonSum)
    pearsonVals[pearsonVals == -1] = sys.maxsize # Replace score with maxsize,  because we have to remove the column

    return pearsonVals



# ----------------------------------------------------------------------------
# Lower values are preferred
# A small interquartile range means that most of the values lie close to each other.
# Zero iqr means that the 3rd and 1st quartile are equals.
# ----------------------------------------------------------------------------
def InterquartileRange(dataset):

    # interpolation : {linear, lower, higher, midpoint, nearest}
    # Specifies the interpolation method to use when the percentile
    # boundaries lie between two data points i and j.
    
    return iqr(dataset, axis=0, interpolation='nearest')



# ----------------------------------------------------------------------------
# Lower intra-class distances must be favored in OCC, in order to make
# positive data closer to each other.
# ----------------------------------------------------------------------------
def IntraClassDistance(dataset, rows, columns):

    # Vector that stores the mean value for each column
    columnMean = dataset.mean(axis=0)
        
    # Computing the Euclidian distance
    distances = np.sqrt(np.sum(np.power(columnMean - dataset, 2), axis=0))

    # Sum of the distance of each column
    globalDistance = np.sum(distances)


    # The following steps simulate the removal of each individual atribute (column)
    # (2 steps)
        
    # (step 1) Distance without tail columns
    best = np.zeros(columns)
    for column in range(columns):
        if (column > 0):
            best[0] += distances[column]
        if (column < columns - 1):
            best[columns - 1] += distances[column]
        

    # (step 2) Distance without intermediate columns
    for column in range(1, columns - 1):
        for i in range(column):
            best[column] += distances[i]
            
        for i in range(column + 1, columns):
            best[column] += distances[i]
        

    # Check if with all columns is better than with less
    check = 0
    for column in range(columns):
        if (best[column] < globalDistance):
            check = 1
            break


    return (np.linspace(1, columns, columns), best)[check == 1]



# ----------------------------------------------------------------------------
def InformationScore(dataset, rows, columns):
    
    dataset = pd.DataFrame(dataset)
    
#    if dataset.shape[0] > 1000: # sample of a large dataset
#        dataset = dataset.iloc[::100]
    dataset = dataset.values

    # STEP 1: Entropy for the dataset with all atributes

    # Create a similarity matrix using RBF kernel function 
    similarityMatrix = constructRBF(dataset)
    newSimilarityMatrix = 0.5 + (similarityMatrix / 2)
    totalEntropy = - calculateEntropy(newSimilarityMatrix)


    # STEP 2: Evaluating each attribute contribution to entropy
    solution = []
    for i in range(columns):

        if (i == 0):
            finalMatrix = dataset[:, 1:columns]

        elif(i == columns - 1):
            finalMatrix = dataset[:, 0:columns - 1]
        else:
            leftMatrix  = dataset[:, 0:i]
            rightMatrix = dataset[:, i + 1:columns]
            finalMatrix = np.concatenate((leftMatrix, rightMatrix), axis=1)
            
        similarityMatrix = constructRBF(finalMatrix)
        newSimilarityMatrix = 0.5 + (similarityMatrix / 2)
        entropy = - calculateEntropy(newSimilarityMatrix)

        solution.append(totalEntropy - entropy)

    return solution


def constructRBF(X):

    return rbf_kernel(X, X)
   

def calculateEntropy(X):
    
    h = X[np.abs(1 - X > 1e-6)]
        
    return np.sum(h * np.log2(h) + (1 - h) * np.log2(1 - h))
