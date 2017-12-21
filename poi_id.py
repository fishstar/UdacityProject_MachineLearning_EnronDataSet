#!/usr/bin/python
# python 2.7

import pickle

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier

from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
import numpy as np



### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 1: Remove outliers
data_dict.pop('TOTAL', 0)


### Task 2: Create new feature(s)
def calc_log(x):
    if x == 'NaN' or float(x) <= 0:
        return 'NaN'
    else:
        return np.log(float(x))

for name, fea in data_dict.items():
    fea['log_salary'] = calc_log(fea['salary'])
    fea['log_bonus'] = calc_log(fea['bonus']) 
    fea['log_total_payments'] = calc_log(fea['total_payments'])  

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Task 3: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'to_messages', 'deferral_payments', 'expenses', 'deferred_income', 'long_term_incentive', \
                 'restricted_stock_deferred', 'shared_receipt_with_poi', 'loan_advances', 'from_messages', 'other', \
                 'director_fees', 'from_poi_to_this_person', 'from_this_person_to_poi', 'restricted_stock', \
                 'exercised_stock_options', 'total_stock_value', \
                 'log_bonus', 'log_salary', 'log_total_payments']

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)



## model evaluation function
def model_test(model, X, y, sss):

    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    X = np.array(X)
    y = np.array(y)
    
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        accuracy_list.append(accuracy_score(y_test, y_pred))
        precision_list.append(precision_score(y_test, y_pred))
        recall_list.append(recall_score(y_test, y_pred))
        f1_list.append(f1_score(y_test, y_pred))
        
    print "accuracy: ", np.mean(accuracy_list)
    print "precision: ", np.mean(precision_list)
    print "recall: ", np.mean(recall_list)
    print "f1: ", np.mean(f1_list)



### Task 4: Try a varity of classifiers
### Task 5: Tune your classifier to achieve better than .3 precision and recall 

steps = [('selector', SelectKBest()),
         ('pca', PCA(random_state=23)),
         ('classifier', GaussianNB())]
pipeline = Pipeline(steps)

parameters = {'selector__k': np.arange(10, 20),
              'pca__n_components': np.arange(2, 11)}

clf_grid = GridSearchCV(pipeline, param_grid=parameters, scoring='f1')
clf_grid.fit(features, labels)
clf = clf_grid.best_estimator_
print "------ model parameters ------"
print clf_grid.best_params_


print "\n------ my model evaluation ------"
sss = StratifiedShuffleSplit(n_splits=1000, test_size=0.2, random_state=23)
model_test(clf, features, labels, sss)


print "\n------ Udacity model evaluation ------"
test_classifier(clf, my_dataset, features_list, folds=1000)


print "\n------ features selected --------"
features_best = clf.named_steps['selector']
features_new = [ features_list[i+1] for i,ele in enumerate(features_best.get_support()) if ele]
features_score = dict()
for i in range(len(features_new)):
    features_score[features_new[i]] = features_best.scores_[i]
print sorted(features_score.items(), key=lambda x: x[1], reverse=True)



### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)