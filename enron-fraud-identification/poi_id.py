#!/usr/bin/python

### Import necessary libraries 
import sys
import pickle
sys.path.append("../tools/")

from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from time import time

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, main
from explorer import data_explorer

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

### Available features
# **financial features**: ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',
# 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options',
# 'other', 'long_term_incentive', 'restricted_stock', 'director_fees']
# **email features**: ['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages',
# 'from_this_person_to_poi', 'shared_receipt_with_poi']

### Label
# **label POI**: [‘poi’]

features_list = ['poi', 'salary', 'deferral_payments', 'total_payments', 'loan_advances',
                 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value',
                 'expenses', 'exercised_stock_options', 'long_term_incentive',
                 'restricted_stock', 'director_fees', 'to_messages', 'from_poi_to_this_person',
                 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']

# Considered all features at first moment
# Removed 'email_address' because has unique value per sample, and it is useless for ML
# Removed 'Other' because is pointless to this ML analysis 

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)
    
### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing

### Data exploration, data wrangling and create new features
my_new_dataset, new_features_list = data_explorer(my_dataset, features_list)

### Extract features and labels from dataset for local testing
#data = featureFormat(my_dataset, features_list, sort_keys = True)
#labels, features = targetFeatureSplit(data)

### Intelligent feature setection ******

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Feature select is performed with SelectKBest where k is selected by GridSearchCV
# Using Stratify for small and minority POI dataset

# Extract features and labels from dataset for local testing
# data = featureFormat(my_new_dataset, new_features_list, sort_keys=True)
# labels, features = targetFeatureSplit(data)
# features_train, features_test, labels_train, labels_test = \
#    train_test_split(features, labels, train_size=.45, stratify=labels)
    
# skbest = SelectKBest(k=10)  # try best value to fit
# sk_transform = skbest.fit_transform(features_train, labels_train)
# indices = skbest.get_support(True)
# print(skbest.scores_)
# for index in indices:
#    print('features: ',(new_features_list[index + 1],' score: ',skbest.scores_[index]))
    
# Selecting top 10 features after some samples of SelectKBest

final_features_list = ['poi', 'salary', 'bonus', 'loan_advances', 'total_stock_value', 
                       'expenses', 'exercised_stock_options', 'total_payments', 
                       'from_poi_to_this_person', 'total_payments_salary_ratio',
                       'bonus_salary_ratio']

final_features_list = ['poi', 'salary', 'bonus', 'loan_advances', 'total_stock_value', 
                       'expenses', 'exercised_stock_options']

# Extract features and labels from dataset for local testing
data = featureFormat(my_new_dataset, final_features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)
    
# Provided to give you a starting point. Try a variety of classifiers.
#from sklearn.linear_model import LogisticRegression
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
#from xgboost.sklearn import XGBClassifier
#from sklearn.neural_network import MLPClassifier

#from sklearn.preprocessing import normalize
#features = normalize(features)

#clf = LogisticRegression()
#clf = KNeighborsClassifier()
#clf = SVC()
#clf = AdaBoostClassifier()
#clf = XGBClassifier()
#clf = MLPClassifier()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
#features_train, features_test, labels_train, labels_test = \
#    train_test_split(features, labels, test_size=0.3, random_state=42)

'''
# Using stratified shuffle split cross validation because of the small size of the dataset
sss = StratifiedShuffleSplit(labels, 500, test_size=0.45, random_state=42)
# Build pipeline
kbest = SelectKBest(k=10)
scaler = MinMaxScaler()
classifier = DecisionTreeClassifier()
pipeline = Pipeline(steps=[('minmax_scaler', scaler), 
                           ('feature_selection', kbest), 
                           ('clf', classifier)])

# Set parameters for decision tree
parameters = [{'clf__criterion': ['gini', 'entropy'],
               'clf__random_state': [None, 46],
               'clf__max_depth': [5, 10, 15]}]

# Get optimized parameters for F1-scoring metrics
cross_val = GridSearchCV(pipeline, 
                         param_grid=parameters, 
                         scoring='recall', cv=sss)
#t0 = time()
cross_val.fit(features, labels)
#print('Classifier tuning: %r', round(time() - t0, 3))
    
print('Best parameters: ', cross_val.best_params_)
clf = cross_val.best_estimator_
#print(clf)
'''

# Using stratified shuffle split cross validation because of the small size of the dataset
sss = StratifiedShuffleSplit(labels, 500, test_size=0.45, random_state=42)
# Build pipeline
kbest = SelectKBest(k=6)
scaler = MinMaxScaler()
classifier = AdaBoostClassifier()
pipeline = Pipeline(steps=[('minmax_scaler', scaler), 
                           ('feature_selection', kbest), 
                           ('clf', classifier)])

# Set parameters for Ada Boost
parameters = [{'clf__base_estimator': [DecisionTreeClassifier(max_depth=10, criterion='gini')],
               'clf__n_estimators': [50, 75],
               'clf__learning_rate': [.5, .7, .8]}]

# Get optimized parameters for F1-scoring metrics
cross_val = GridSearchCV(pipeline, 
                         param_grid=parameters, 
                         scoring='recall', cv=sss)
#t0 = time()
cross_val.fit(features, labels)
#print('Classifier tuning: %r', round(time() - t0, 3))
    
print('Best parameters: ', cross_val.best_params_)
clf = cross_val.best_estimator_
#print(clf)


clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=10, criterion='gini'), 
                         n_estimators=75, 
                         learning_rate=.7)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_new_dataset, final_features_list)
main()