#!/usr/bin/python

""" 
    Support functions to Enron project to help explore data.
"""
import matplotlib  
matplotlib.use('TkAgg')   
import matplotlib.pyplot as plt  

from collections import Counter
from feature_format import featureFormat, targetFeatureSplit



def scatter_plot(dataset, x_feature, y_feature):
    """ Create scatter plot, save to local path
    """
    features = ['poi', x_feature, y_feature]
    data = featureFormat(dataset, features)

    for point in data:
        x = point[1] # feature
        y = point[2] # feature        
        if point[0]: # label
            plt.scatter(x, y, color="r", marker="*") # Is a POI
        else:
            plt.scatter(x, y, color='b', marker=".") # Is a non-POI
    plt.xlabel(x_feature, fontsize=11)
    plt.ylabel(y_feature, fontsize=11)
    pic = x_feature + "_" + y_feature + '.png'
    plt.tight_layout()
    plt.savefig(pic, transparent = True)



def poi_missing_email_explorer(my_dataset):
    """ Find total count and values of POI with missing or no to/from email information.
    Args:
        my_dataset: dataset of Enron.
    Returns:
        POIs with missing or no to/from email information.
    """

    poi_missing_emails = []
    non_poi_missing_emails = []
            
    for person in my_dataset.keys():
        if (my_dataset[person]['to_messages'] == 'NaN' and my_dataset[person]['from_messages'] == 'NaN') or \
           (my_dataset[person]['to_messages'] == 0 and my_dataset[person]['from_messages'] == 0):
                if my_dataset[person]["poi"]:
                    poi_missing_emails.append(person)
                else:
                    non_poi_missing_emails.append(person)

    return poi_missing_emails, non_poi_missing_emails



def new_features_creator(my_dataset, features_list):
    """ Create the following features in the dataset: bonus_salary_ratio, 
    total_payments_salary_ratio and total_stock_value_ratio.
    Args:
        my_dataset: dataset of Enron.
        features_list: all features used in the analysis.
    Returns:
        Enron dataset with new features.
    """
    for person in my_dataset.keys():
        person_salary = float(my_dataset[person]['salary'])
        
        #bonus/salary
        person_bonus = float(my_dataset[person]['bonus'])        
        if person_salary > 0 and person_bonus > 0:
            my_dataset[person]['bonus_salary_ratio'] = my_dataset[person]['bonus'] / my_dataset[person]['salary']
        else:
            my_dataset[person]['bonus_salary_ratio'] = 0
        
        #total_payment/salary
        person_bonus = float(my_dataset[person]['total_payments'])        
        if person_salary > 0 and person_bonus > 0:
            my_dataset[person]['total_payments_salary_ratio'] = my_dataset[person]['total_payments'] / my_dataset[person]['salary']
        else:
            my_dataset[person]['total_payments_salary_ratio'] = 0
            
        #total_stock_value/salary
        person_bonus = float(my_dataset[person]['total_stock_value'])        
        if person_salary > 0 and person_bonus > 0:
            my_dataset[person]['total_stock_value_ratio'] = my_dataset[person]['total_stock_value'] / my_dataset[person]['salary']
        else:
            my_dataset[person]['total_stock_value_ratio'] = 0

    # Add new feature to features_list
    features_list.extend(['bonus_salary_ratio', 'total_payments_salary_ratio', 'total_stock_value_ratio'])
    
    return my_dataset, features_list


def data_explorer(my_dataset, features_list):
    """ Process the data, exploring it and make the necessary adjusts on data
        to ML analysis.
    Args:
        my_dataset: dataset of Enron.
        features_list: all features used in the analysis.
    Returns:
        The dataset updated to ML analysis.
    """
    
    ### Extract features and labels from dataset for local testing and remove NaN and all zeroes
    data = featureFormat(my_dataset, features_list, remove_NaN = True, remove_all_zeroes = True, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    
    ### Take a look in the dataset to verify the % of POI and non-POI samples
    # Number of samples
    #print("Number of samples: ", data.shape[0])
    # Number of POI and non-POI
    #print("Count of non-POI(0.0) and POI(1.0)", Counter(labels))
    
    ### Verify POI with missing or no to/from email information
    poi_missing_emails, non_poi_missing_emails = poi_missing_email_explorer(my_dataset)
    #print("POIs with zero or missing to/from email messages in dataset: ", len(poi_missing_emails))
    #print("They are: ", poi_missing_emails)
    #print("Non-POIs with zero or missing to/from email messages in dataset: ", len(non_poi_missing_emails))
    #print("They are: ", non_poi_missing_emails)
    
    ### Some odd names in dataset, brute force
    # print("All names: ", my_dataset.keys())
    # found ['TOTAL', 'THE TRAVEL AGENCY IN THE PARK']
    
    high_salaries = []
    for person in my_dataset.keys():
        if my_dataset[person]["bonus"] != 'NaN' and my_dataset[person]["salary"] != 'NaN':
            if int(my_dataset[person]["bonus"]) > 5000000 or int(my_dataset[person]["salary"]) > 1000000:
                high_salaries.append(person)
                # print(person, " Salary: ", my_dataset[person]["salary"]) 
                # ['FREVERT MARK A', 'TOTAL', 'BELDEN TIMOTHY N', 'SKILLING JEFFREY K', 'LAY KENNETH L', 'LAVORATO JOHN J']

    
    # Remove the outliers
    my_dataset.pop('TOTAL')
    my_dataset.pop('THE TRAVEL AGENCY IN THE PARK')
    #my_dataset.pop('FREVERT MARK A')
    #my_dataset.pop('BELDEN TIMOTHY N')
    #my_dataset.pop('SKILLING JEFFREY K')
    #my_dataset.pop('LAY KENNETH L')
    #my_dataset.pop('LAVORATO JOHN J')
    
    # Print some scatter plots to the report
    #scatter_plot(my_dataset, 'salary', 'total_payments')
    #scatter_plot(my_dataset, 'salary', 'bonus')
    #scatter_plot(my_dataset, 'salary', 'total_stock_value')
    #scatter_plot(my_dataset, 'salary', 'expenses')
    #scatter_plot(my_dataset, 'salary', 'director_fees')
    
    # Crate new features
    # 'bonus_salary_ratio', 'total_payments_salary_ratio', 'total_stock_value_ratio'
    my_dataset, features_list = new_features_creator(my_dataset, features_list)

    return my_dataset, features_list