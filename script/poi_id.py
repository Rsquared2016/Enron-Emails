import sys
import pickle
from copy import copy
sys.path.append("../tools/")
import data_tools

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
financial_features = ['salary', 
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
                      'director_fees'] 

email_features= ['to_messages', 
                #'email_address',  # remit email address label
                 'from_poi_to_this_person', 
                 'from_messages', 
                 'from_this_person_to_poi', 
                 'shared_receipt_with_poi'] 

POI_label= ['poi'] 

feature_list = POI_label +  financial_features + email_features # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
    
### Explore dataset 
total_data_point, total_valid_data_point, number_of_poi, number_of_features, \
    features_list,  valid_data_by_feature, valid_data_by_person = data_tools.explore_data(data_dict)
print "Total data point in enron dataset:", total_data_point, "\n"
print "Number of valid data point in enron dataset:", total_valid_data_point, "\n"
print "Number of poi in enron dataset:", number_of_poi, "\n"
print "Number of features in enron dataset:", number_of_features, "\n"     

### Task 2: Remove outliers
outliers = ['TOTAL', 'THE TRAVEL AGENCY IN THE PARK', 'LOCKHART EUGENE E']
for outlier in outliers:
    data_dict.pop(outlier, 0)
        
### Store to my_dataset for easy export below.
my_dataset = copy(data_dict)
my_feature_list = copy(feature_list)
 
### Task 3: Create new feature(s)
data_tools.add_feature_financial_sum(my_dataset, my_feature_list)
data_tools.add_feature_message_proportion_with_poi(my_dataset, my_feature_list)

### Get K-best features 
k_features = 10
k_best_features = data_tools.get_k_best(my_dataset, my_feature_list, k_features)
my_feature_list = POI_label + k_best_features.keys()

### Print selected feature names and score
print "{0} selected features: {1}\n".format(len(my_feature_list) - 1, my_feature_list[1:])


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, my_feature_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Rescale features via StandardScaler transformation
from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
features = scaler.fit_transform(features)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
### Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
nb_clf = GaussianNB()

### Support Vector Machine (SVM) Classifier
from sklearn.svm import SVC
s_clf = SVC(kernel='rbf', C=1000, class_weight='balanced')

### Nearest Neighbors Classifer
from sklearn.neighbors import KNeighborsClassifier
knn_clf =  KNeighborsClassifier(n_neighbors=3)

### Stochastic Gradient Descent - Logistic Regression
from sklearn.linear_model import SGDClassifier
g_clf = SGDClassifier(loss='log', class_weight = "balanced")

### K-means Clustering
from sklearn.cluster import KMeans
k_clf = KMeans(n_clusters=2, tol=0.001)

### Random Forest
from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier()


### parameter optimization 
from sklearn.model_selection import GridSearchCV

### tune parameters for K-means Clustering; uncomment to run
########################################
### trails = 1
### parameters = {"tol": [1e-15, 1e-10, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]}
### k_clf = GridSearchCV(k_clf, parameters) 
### print "(score, precision, recall) using optimized Stochastic Gradient Descent:",\
###    data_tools.evaluate_clf(k_clf, features, labels, trials, test_size, is_pca, n_components), "\n"
### k_clf = k_clf.best_estimator_


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
### Algorithm evaluations
is_pca = False  
trials = 1000
test_size = 0.3
n_components = 5

print "(score, precision, recall) using GaussianNB:",\
    data_tools.evaluate_clf(nb_clf, features, labels, trials, test_size,  is_pca, n_components), "\n"
print "(score, precision, recall) using Support Vector Machines:",\
    data_tools.evaluate_clf(s_clf, features, labels, trials, test_size, is_pca, n_components), "\n"
print "(score, precision, recall) using Nearest Neighbors:",\
    data_tools.evaluate_clf(nb_clf, features, labels, trials, test_size,  is_pca, n_components), "\n"
print "(score, precision, recall) using Stochastic Gradient Descent:",\
    data_tools.evaluate_clf(g_clf, features, labels, trials, test_size, is_pca, n_components), "\n"
print "(score, precision, recall) using K-means:",\
    data_tools.evaluate_clf(k_clf, features, labels, trials, test_size, is_pca, n_components), "\n"
print "(score, precision, recall) using Random Forests:",\
    data_tools.evaluate_clf(rf_clf, features, labels, trials, test_size, is_pca, n_components), "\n"


# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

clf = nb_clf
    
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)

pickle.dump(clf, open("my_classifier.pkl", "w"))
pickle.dump(my_dataset, open("my_dataset.pkl", "w"))
pickle.dump(my_feature_list, open("my_feature_list.pkl", "w"))