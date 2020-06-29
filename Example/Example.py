# Example created by Miryam Elizabeth Villa-Pérez


import pandas as pd
import m_ockra

from sklearn.metrics import roc_auc_score


def load_datasets(train_file, test_file):
        
    print('Reading datasets...')
    
    train = pd.read_csv(train_file, header=None)
    test = pd.read_csv(test_file, header=None)
        
    return train, test


# Reading files
train, test = load_datasets('training.csv', 'testing.csv')


# Split train, test
rows, columns = train.shape

X_train = train.iloc[:, :columns-1]
y_train = train.values[:, -1]

X_test = test.iloc[:, :columns-1]
y_test = test.values[:, -1]


# Training phase
# ---------------------------
# We use by default the parameters of the article:
# 'm-OCKRA: An Efficient One-Class Classifier for Personal Risk Detection, Based on Weighted Selection of Attributes' (https://doi.org/10.1109/ACCESS.2020.2976947)
# classifier_count=50, bootstrap_sample_percent(F)=0.4, mros_percent(RS)=0.4, method='Majority', distance_metric='chebyshev'
# ---------------------------
# If we want other parameters...
# m_OCKRA(classifier_count=25, bootstrap_sample_percent=0.2, mros_percent=0.1, method='Borda', distance_metric='euclidean')

print('Training classifier...')
clf = m_ockra.m_OCKRA()
clf.fit(X_train, y_train)


# Testing phase
print('Classifier trained!')
print('Testing classifier...')
        
y_pred = clf.score_samples(X_test)
auc = roc_auc_score(y_test,  y_pred)
print(f'Testing AUC: {auc if auc > .5 else 1 - auc}')
