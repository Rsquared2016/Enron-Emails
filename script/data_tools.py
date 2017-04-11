import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../tools/")

from feature_format import featureFormat
from feature_format import targetFeatureSplit

from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score

def explore_data(data_dict):
    ### This function returns the number of non-NaN values for each feature ###
    total_data_point = 0
    total_valid_data_point = 0
    number_of_poi = 0
    features = []
    valid_data_by_feature = {}
    valid_data_by_person = {}
    
    for name, value in data_dict.items():
        total_data_point += len(value)     ### total number of data points
        valid_point_individual = 0
        for feature in value:
            
            if feature not in features and feature != "poi" and feature != "email_address":
                features.append(feature)    ### feature name in enron dataset
                valid_data_by_feature[feature] = 0
            if feature == "poi":
                if value[feature] == True:
                    number_of_poi += 1    ### count number of POIs      
            if value[feature] != "NaN":
                total_valid_data_point += 1    ### count number of valid data point for each feature
                valid_point_individual += 1
                if feature != "poi" and feature != "email_address":
                    valid_data_by_feature[feature] +=1
            
        valid_data_by_person[name] = valid_point_individual
    return (total_data_point, total_valid_data_point, number_of_poi, len(features),\
        features, valid_data_by_feature, valid_data_by_person)

def visualize_data(data_dict, feature_x, feature_y):
    ### This funciton generates a plot of feature y vs feature x, colors poi ###

    data = featureFormat(data_dict, [feature_x, feature_y, 'poi'])

    for point in data:
        x = point[0]
        y = point[1]
        poi = point[2]
        color = 'red' if poi else 'blue'
        plt.scatter(x, y, color=color)
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.show()

def add_feature_financial_sum(data_dict, features_list):
    ### add total value of salary, bonus and stock ###

    fields = ['total_stock_value', 'salary', 'bonus']
    for record in data_dict:
        person = data_dict[record]
        is_valid = True
        for field in fields:
            if person[field] == 'NaN':
                is_valid = False
        if is_valid:
            person['financial_sum'] = sum([person[field] for field in fields])
        else:
             ### assign value to NaN if having invalid data
            person['financial_sum'] = 'NaN'
    features_list += ['financial_sum']

def add_feature_message_proportion_with_poi(data_dict, features_list):
    ### calculates proportion of message interacting with pois
    ### adds feature 'message_proportion_with_poi' to data dict
    fields = ['to_messages', 'from_messages',
              'from_poi_to_this_person', 'from_this_person_to_poi']
    for record in data_dict:
        person = data_dict[record]
        is_valid = True
        for field in fields:
            if person[field] == 'NaN':
                is_valid = False
        if is_valid:
            total_messages = person['to_messages'] + person['from_messages']
            poi_messages = person['from_poi_to_this_person'] + person['from_this_person_to_poi']
            person['message_proportion_with_poi'] = float(poi_messages) / total_messages
        else:
             ### assign value to NaN if having invalid data
            person['message_proportion_with_poi'] = 'NaN' 
    features_list += ['message_proportion_with_poi']


def get_k_best(data_dict, features_list, k_features):
    ### Use scikit-learn's SelectKBest for features selection
    ### Return dict where keys=features, values=scores
    
    data = featureFormat(data_dict, features_list)
    labels, features = targetFeatureSplit(data)

    k_best = SelectKBest(k=k_features)
    k_best.fit(features, labels)
    scores = k_best.scores_
    feature_and_score = sorted(zip(features_list[1:], scores), key = lambda x: x[1], reverse = True)
    k_best_features = dict(feature_and_score[:k_features])
    print "{0} best features and scores: {1}\n".format(k_features, feature_and_score[:k_features])
    return k_best_features


def evaluate_clf(clf, features, labels, trials, test_size, is_pca, n_components): 
    ### evaluation function for machine learning algorithm: 
    ### returns accuracy_score, precision and recall

    score = []
    precision = []
    recall = []
    # run 1000 trials and get mean values
    for trial in range(trials): 
        features_train, features_test, labels_train, labels_test = \
            train_test_split(features, labels, test_size = test_size, random_state = 42)
        if is_pca:
            pca = PCA(n_components = n_components, whiten=True).fit(features_train)
            features_train = pca.transform(features_train)
            features_test = pca.transform(features_test)
        clf.fit(features_train, labels_train)
        pred = clf.predict(features_test)
        score.append(accuracy_score(pred, labels_test))
        precision.append(precision_score(pred, labels_test))
        recall.append(recall_score(pred, labels_test))

    #print "precision: {}".format(np.mean(precision))
    #print "recall:    {}".format(np.mean(recall))

    return np.mean(score), np.mean(precision), np.mean(recall)


if __name__ == "__main__":
    data_dict = pickle.load(open("final_project_dataset.pkl", "r"))
    features_list = ['poi',
                     'salary', 
                      'deferral_payments', 
                      'total_payments', 
                      'loan_advances', 
                      'bonus', 
                      'restricted_stock_deferred', 
                      'deferred_income', 
                      'total_stock_value', 
                      'expenses', 
                      'exercised_stock_options', 
                      'other', 
                      'long_term_incentive', 
                      'restricted_stock', 
                      'director_fees',
                     'from_messages',
                     'from_poi_to_this_person',
                     'from_this_person_to_poi',
                     'shared_receipt_with_poi',
                     'to_messages']

    k_best = get_k_best(data_dict, features_list, 10)
