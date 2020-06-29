# m-OCKRA

[m-OCKRA](https://ieeexplore.ieee.org/document/9018030/)<sup>1</sup> is a new ensemble of one-class classifiers based on weighted attribute projection using filters. It has been specially designed for the personal risk detection problem using the [PRIDE dataset](https://www.sciencedirect.com/science/article/pii/S002002551630576X)<sup>2</sup>.

m-OCKRA outperforms its predecessor version [OCKRA](https://www.mdpi.com/1424-8220/16/10/1619)<sup>3</sup>, by speeding-up training execution time and by preserving its classification performance.


## Usage

A practical example about how to use OCKRA can be seen in [Example](https://github.com/Miel15/m-OCKRA/tree/master/Example).


## Requirements

Python 3, Scikit-learn, Pandas, Numpy


## User Guide

```
class m-OCKRA(classifier_count=50, bootstrap_sample_percent=0.4, mros_percent=0.4, method='Majority', distance_metric='chebyshev', use_bootstrap_sample_count=True, user_threshold=95)
```

### Parameters

 - **classifier_count : int, default=50**
Number of classifiers in the ensemble.

- **bootstrap_sample_percent : float, default=0.4**
The fraction of training dataset to bootstrap (a number between 0  and 1).

- **mros_percent : float, default=0.4**
The most representative objects (MROs) selected by random sampling without replacement.

- **method : {'Mean', 'Majority', 'Borda'}, default='Majority'**
Aggregation method by the weighted feature selection.

- **distance_metric : {'euclidean', 'manhattan', 'chebyshev'}, default='chebyshev'**
The metric to use when calculating distance between instances and the MROs.


### Methods

```
fit(self, X, y)
```
Training of the OCC algorithm.

**Parameters**

 - **X: *array-like* of shape (n_samples, n_features)**
Set of samples, where *n_samples* is the number of instances and *n_features* is the number of features. 
- **y: *Ignored* array-like of shape (n_samples, 1)**
Not used, present for API consistency by convention.

**Return**
- **self: *object*** 

```
predict(self, X)
```
Perform classification on samples in *X*. For a one-class model, +1 or -1 is returned.

**Parameters**

 - **X: *array-like* of shape (n_samples, n_features)**
Set of samples, where *n_samples* is the number of instances and *n_features* is the number of features.

**Return**
- **y_pred: *ndarray* of shape (n_samples, )**
Class labels for samples in *X*.


```
score_samples(self, X)
```
Raw scoring function of the samples. Compute the similarity value.

**Parameters**

 - **X: *array-like* of shape (n_samples, n_features)**
The data array.

**Return**
- **y_pred: *ndarray* of shape (n_samples, )**
Class labels for samples in *X*.


## Algorithm

### Training phase
   * **Input:**
        * *T*: training dataset;
        * *N*: number of classifiers in the ensemble;
        * *F*: fraction of training dataset to bootstrap;
        * *RS<sub>%</sub>*: MROs (Most Representative Objects) percentage.
   * **Local Variables:**
        * *W*: list of attributes weights;
        * *SelectedFeatures*: randomly-selected features;
        * *T'*: projected dataset *T* over the randomly-selected features;
        * *X*: a sample with replacement of *F* objects from *T'*;
        * *MROs*: a sample without replacement of *RS<sub>%</sub>* objects from *X'*;
        * *δ<sub>i</sub>*: classiﬁer threshold.
   * **Output:**
        * *P*: the set of classifiers parameters (randomly-selected features, the MROs and the threshold).
   * **Start**:
        1. Set *P* initially empty; i.e., *P* ← {}
        2. *W* ← ComputeAttributesWeights(*T*)
        2. **for** *i*= 1..*N* **do**:
	        1. *SelectedFeatures*  ← RandomWeightedAttributes(*T*)
	        2. *T'* ← Project(*T*, *SelectedFeatures*)
	        3. *X* ← Bootstrap(*F*, *T'*)
          4. *MROs*  ← SampleWithoutReplacement(*RS<sub>%</sub>*, *X*)
	        4. *δ<sub>i</sub>* ← SumAttributesWeights(*W*, *SelectedFeatures*)
	        5. *P* ← *P* U { (*SelectedFeatures*, *Centres*, *δ<sub>i</sub>* ) }
        3. **end for**
        4. **return** *P*


### Classification phase
   * **Input:**
        * *O*: object to be classified;
        * *P*: the set of parameters computed in the training phase.
   * **Local Variables:**
        * *O'*: projected object *O* over the randomly-selected features;
        * *d<sub>min</sub>*:  the nearest MRO to *O′* (smallest distance between the selecting MRO and the object *O'*).
   * **Output:**
        * *s*: similarity value (zero indicates an anormal behaviour and one represents normal behavior).
   * **Start:**
        1. Let *s* ← 0 be the similarity value computed by the classifiers
        2. **for each** (*Features<sub>i</sub>*, *MROs<sub>i</sub>*, *δ<sub>i</sub>*) **in** *P* **do**:
            1. *O'* ← Project(*O*, *Features<sub>i</sub>*)
            2. *d<sub>min</sub>* ← min (Distance(O', *MROs<sub>i</sub>*))
            3. *s* ← *s* + *e*^(-0.5(*d<sub>min</sub>* ∕ *δ<sub>i</sub>* )^2 )
        3. **end for**
        4. **return** *s* / | *P* |


## References

 1. *For more information about m-OCKRA, please read:*
	 M. E. Villa-Pérez and L. A. Trejo, ["m-OCKRA: An Efficient One-Class Classifier for PersonalRisk Detection, Based on Weighted Selection of Attributes"](https://ieeexplore.ieee.org/document/9018030), _IEEE Access_ vol. 8, pp. 41749-41763, Feb. 2020.
	 
 2. *For more information about the PRIDE dataset, please read:*
	A. Y. Barrera-Animas, L. A. Trejo, M. A. Medina-Pérez, R. Monroy, J. B. Camiña and F. Godínez, ["Online personal risk detection based on behavioural and physiological patterns"](https://www.sciencedirect.com/science/article/pii/S002002551630576X), _Information Sciences_, vol. 384, pp. 281-297, Apr. 2017.

 3. *For more information about OCKRA, please read:*
	J. Rodríguez, A. Barrera-Animas, L. Trejo, M. Medina-Pérez and R. Monroy, ["Ensemble of one-class classifiers for personal risk detection based on wearable sensor data"](https://www.mdpi.com/1424-8220/16/10/1619), _Sensors_, vol. 16, no. 10, pp. 1619, Sep. 2016.

