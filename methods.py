# --------------------------------------------------------------------------------------------
# Aggregation metrics
# --------------------------------------------------------------------------------------------


import numpy as np
import pandas as pd
from scipy.stats import rankdata
from collections import Counter


def getMean(rank_list):

    return rankdata(np.mean(rank_list, axis=0), method='ordinal')
    

def getMajority(rank_list):
    
    dataset = pd.DataFrame(rank_list)

    values = []
    for i in range(dataset.shape[1]):
        column = sorted(dataset.loc[: , i])
        count = Counter(column).most_common()[0][0]
        values.append(count)

    return rankdata(values, method='ordinal')
        

def getBorda(rank_list):

    dataset = pd.DataFrame(rank_list)
    lists = dataset.values

    scores = {}

    for l in lists:
        for idx, elem in enumerate(reversed(l)):
            if not elem in scores:
                scores[elem] = 0
                scores[elem] += idx
        
    result = sorted(scores.keys(), key=lambda elem: scores[elem], reverse=True)

    return np.array(result)
