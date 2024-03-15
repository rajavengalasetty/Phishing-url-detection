# Step 1 import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import confusion_matrix




# Step 2 read the csv files and create pandas dataframes
legitimate_df = pd.read_csv("structured_data_legitimate.csv")
phishing_df = pd.read_csv("structured_data_phishing.csv")

# Step 3 combine legitimate and phishing dataframes, and shuffle
df = pd.concat([legitimate_df, phishing_df], axis=0)

#column to list
df.columns.tolist()

# Analyzing the shape of the data
df.shape

# check for missing values:
df.isnull().sum()

#Heatmap for missing values
missing = df.isnull()
plt.figure(figsize=(8, 6))
sns.heatmap(missing, cmap='viridis', cbar=True)
plt.title('Missing Values Heatmap')
plt.show()

#data information
df.info()

#checking duplicate values
df.nunique(axis=0, dropna=False)

# describing the data
df.describe()

#Shuffling the data
df = df.sample(frac=1)
df.head()


# Step 4 remove'url' and remove duplicates, then we can create X and Y for the models, Supervised Learning
df = df.drop('URL', axis=1)

# Removing duplicates
df = df.drop_duplicates()
df.head()

#Pairplot1
selected_columns=['has_title',
 'has_input',
 'has_button',
 'has_image',
 'has_submit']
sns.pairplot(df[selected_columns])
plt.show()

#Pairplot2
selected_columns=['has_link',
 'has_password',
 'has_email_input',
 'has_hidden_element',
 'has_audio']
sns.pairplot(df[selected_columns])
plt.show()


X = df.drop('label', axis=1)
Y = df['label']


# Step 5 split data to train and test
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=10)


# Step 6 create a ML model using sklearn
svm_model = svm.LinearSVC()

# Random Forest
rf_model = RandomForestClassifier(n_estimators=60)

# Decision Tree
dt_model = tree.DecisionTreeClassifier()

# AdaBoost
ab_model = AdaBoostClassifier()

# Gaussian Naive Bayes
nb_model = GaussianNB()

# Neural Network
nn_model = MLPClassifier(alpha=1)

# KNeighborsClassifier
kn_model = KNeighborsClassifier()

# Step 7 train the model
svm_model.fit(x_train, y_train)


# Step 8 make some predictions using test data
predictions = svm_model.predict(x_test)


# Step 9 create a confusion matrix and tn, tp, fn , fp
tn, fp, fn, tp = confusion_matrix(y_true=y_test, y_pred=predictions).ravel()




# Train a classifier (Random Forest as an example)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(x_train, y_train)

# Predict probabilities for the positive class
y_scores = clf.predict_proba(x_test)[:, 1]

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()





# Train a classifier (Random Forest as an example)
clf = DecisionTreeClassifier( random_state=42)
clf.fit(x_train, y_train)

# Predict probabilities for the positive class
y_scores = clf.predict_proba(x_test)[:, 1]

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

#heatmap for confusion matrix values
def plot_confusion_matrix(confusion_matrix):
    plt.figure(figsize=(4, 4))
    sns.set(font_scale=1.2)
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['PN', 'PP'],
                yticklabels=['AN', 'AP'])
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix')
    plt.show()
conf_matrix = np.array([[tn, fp], [fn, tp]])

plot_confusion_matrix(conf_matrix)

# Step 10 calculate accuracy, precision and recall scores
import math
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
misclassification = (fp + fn)/(tp + tn + fp + fn)
sensitivity = tp/(tp + fn)
specificity = tn/(tn + fp)
f1_score = (2*precision*recall) / (precision + recall)
false_positive_rate=fp / (fp+tn)
false_negative_rate=fn / (fn+tp)
Matthews_Correlation_Coefficient = ((tp*tn) - (fp*fn)) / math.sqrt((tp+fp)*(tp+fn)*(tn+fp))

print("accuracy --> ", accuracy)
print("precision --> ", precision)
print("recall --> ", recall)
print("misclassification --->", misclassification)
print("sensitivity --->", sensitivity)
print("specificity --->",specificity)

# K-fold cross validation, and K = 5
K = 5
total = X.shape[0]
index = int(total / K)

# 1
X_1_test = X.iloc[:index]
X_1_train = X.iloc[index:]
Y_1_test = Y.iloc[:index]
Y_1_train = Y.iloc[index:]

# 2
X_2_test = X.iloc[index:index*2]
X_2_train = X.iloc[np.r_[:index, index*2:]]
Y_2_test = Y.iloc[index:index*2]
Y_2_train = Y.iloc[np.r_[:index, index*2:]]

# 3
X_3_test = X.iloc[index*2:index*3]
X_3_train = X.iloc[np.r_[:index*2, index*3:]]
Y_3_test = Y.iloc[index*2:index*3]
Y_3_train = Y.iloc[np.r_[:index*2, index*3:]]

# 4
X_4_test = X.iloc[index*3:index*4]
X_4_train = X.iloc[np.r_[:index*3, index*4:]]
Y_4_test = Y.iloc[index*3:index*4]
Y_4_train = Y.iloc[np.r_[:index*3, index*4:]]

# 5
X_5_test = X.iloc[index*4:]
X_5_train = X.iloc[:index*4]
Y_5_test = Y.iloc[index*4:]
Y_5_train = Y.iloc[:index*4]


# X and Y train and test lists
X_train_list = [X_1_train, X_2_train, X_3_train, X_4_train, X_5_train]
X_test_list = [X_1_test, X_2_test, X_3_test, X_4_test, X_5_test]

Y_train_list = [Y_1_train, Y_2_train, Y_3_train, Y_4_train, Y_5_train]
Y_test_list = [Y_1_test, Y_2_test, Y_3_test, Y_4_test, Y_5_test]


def calculate_measures(tn, tp, fn, fp):
    accuracy = (tn + tp) / (tn + tp + fn + fp)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    false_positive_rate = fp / (fp+tn)
    false_negative_rate = fn / (fn+tp)
    misclassification = (fp + fn)/(tp + tn + fp + fn)
    f1_score = (2*precision*recall) / (precision + recall)
    return accuracy, precision, recall, sensitivity, specificity, false_positive_rate, false_negative_rate, misclassification, f1_score

rf_accuracy_list, rf_precision_list, rf_recall_list ,rf_sensitivity_list, rf_specificity_list, rf_false_positive_rate_list, rf_false_negative_rate_list, rf_misclassification_list, rf_f1_score_list= [], [], [], [], [], [], [], [], []
dt_accuracy_list, dt_precision_list, dt_recall_list, dt_sensitivity_list, dt_specificity_list, dt_false_positive_rate_list, dt_false_negative_rate_list, dt_misclassification_list, dt_f1_score_list= [], [], [], [], [], [], [], [], []
ab_accuracy_list, ab_precision_list, ab_recall_list, ab_sensitivity_list, ab_specificity_list, ab_false_positive_rate_list, ab_false_negative_rate_list, ab_misclassification_list, ab_f1_score_list = [], [], [], [], [], [], [], [], []
svm_accuracy_list, svm_precision_list, svm_recall_list, svm_sensitivity_list, svm_specificity_list, svm_false_positive_rate_list, svm_false_negative_rate_list, svm_misclassification_list, svm_f1_score_list = [], [], [], [], [], [], [], [], []
nb_accuracy_list, nb_precision_list, nb_recall_list, nb_sensitivity_list, nb_specificity_list, nb_false_positive_rate_list, nb_false_negative_rate_list, nb_misclassification_list, nb_f1_score_list = [], [], [], [], [], [], [], [], []
nn_accuracy_list, nn_precision_list, nn_recall_list, nn_sensitivity_list, nn_specificity_list, nn_false_positive_rate_list, nn_false_negative_rate_list, nn_misclassification_list, nn_f1_score_list = [], [], [], [], [], [], [], [], []
kn_accuracy_list, kn_precision_list, kn_recall_list, kn_sensitivity_list, kn_specificity_list, kn_false_positive_rate_list, kn_false_negative_rate_list, kn_misclassification_list, kn_f1_score_list = [], [], [], [], [], [], [], [], []



for i in range(0, K):
    # ----- RANDOM FOREST ----- #
    rf_model.fit(X_train_list[i], Y_train_list[i])
    rf_predictions = rf_model.predict(X_test_list[i])
    tn, fp, fn, tp = confusion_matrix(y_true=Y_test_list[i], y_pred=rf_predictions).ravel()
    rf_accuracy, rf_precision, rf_recall, rf_sensitivity,rf_specificity, rf_false_positive_rate, rf_false_negative_rate, rf_misclassification, rf_f1_score = calculate_measures(tn, tp, fn, fp)
    rf_accuracy_list.append(rf_accuracy)
    rf_precision_list.append(rf_precision)
    rf_recall_list.append(rf_recall)
    rf_sensitivity_list.append(rf_sensitivity)
    rf_specificity_list.append(rf_specificity)
    rf_false_positive_rate_list.append(rf_false_positive_rate)
    rf_false_negative_rate_list.append(rf_false_negative_rate)
    rf_misclassification_list.append(rf_misclassification)
    rf_f1_score_list.append(rf_f1_score)

    # ----- DECISION TREE ----- #
    dt_model.fit(X_train_list[i], Y_train_list[i])
    dt_predictions = dt_model.predict(X_test_list[i])
    tn, fp, fn, tp = confusion_matrix(y_true=Y_test_list[i], y_pred=dt_predictions).ravel()
    dt_accuracy, dt_precision, dt_recall, dt_sensitivity, dt_specificity, dt_false_positive_rate, dt_false_negative_rate, dt_misclassification, dt_f1_score = calculate_measures(tn, tp, fn, fp)
    dt_accuracy_list.append(dt_accuracy)
    dt_precision_list.append(dt_precision)
    dt_recall_list.append(dt_recall)
    dt_sensitivity_list.append(dt_sensitivity)
    dt_specificity_list.append(dt_specificity)
    dt_false_positive_rate_list.append(dt_false_positive_rate)
    dt_false_negative_rate_list.append(dt_false_negative_rate)
    dt_misclassification_list.append(dt_misclassification)
    dt_f1_score_list.append(dt_f1_score)

    # ----- SUPPORT VECTOR MACHINE ----- #
    svm_model.fit(X_train_list[i], Y_train_list[i])
    svm_predictions = svm_model.predict(X_test_list[i])
    tn, fp, fn, tp = confusion_matrix(y_true=Y_test_list[i], y_pred=svm_predictions).ravel()
    svm_accuracy, svm_precision, svm_recall, svm_sensitivity, svm_specificity, svm_false_positive_rate, svm_false_negative_rate, svm_misclassification, svm_f1_score = calculate_measures(tn, tp, fn, fp)
    svm_accuracy_list.append(svm_accuracy)
    svm_precision_list.append(svm_precision)
    svm_recall_list.append(svm_recall)
    svm_sensitivity_list.append(svm_sensitivity)
    svm_specificity_list.append(svm_specificity)
    svm_false_positive_rate_list.append(svm_false_positive_rate)
    svm_false_negative_rate_list.append(svm_false_negative_rate)
    svm_misclassification_list.append(svm_misclassification)
    svm_f1_score_list.append(svm_f1_score)

    # ----- ADABOOST ----- #
    ab_model.fit(X_train_list[i], Y_train_list[i])
    ab_predictions = ab_model.predict(X_test_list[i])
    tn, fp, fn, tp = confusion_matrix(y_true=Y_test_list[i], y_pred=ab_predictions).ravel()
    ab_accuracy, ab_precision, ab_recall,ab_sensitivity, ab_specificity, ab_false_positive_rate, ab_false_negative_rate, ab_misclassification, ab_f1_score = calculate_measures(tn, tp, fn, fp)
    ab_accuracy_list.append(ab_accuracy)
    ab_precision_list.append(ab_precision)
    ab_recall_list.append(ab_recall)
    ab_sensitivity_list.append(ab_sensitivity)
    ab_specificity_list.append(ab_specificity)
    ab_false_positive_rate_list.append(ab_false_positive_rate)
    ab_false_negative_rate_list.append(ab_false_negative_rate)
    ab_misclassification_list.append(ab_misclassification)
    ab_f1_score_list.append(ab_f1_score)

    # ----- GAUSSIAN NAIVE BAYES ----- #
    nb_model.fit(X_train_list[i], Y_train_list[i])
    nb_predictions = nb_model.predict(X_test_list[i])
    tn, fp, fn, tp = confusion_matrix(y_true=Y_test_list[i], y_pred=nb_predictions).ravel()
    nb_accuracy, nb_precision, nb_recall, nb_sensitivity, nb_specificity, nb_false_positive_rate, nb_false_negative_rate, nb_misclassification, nb_f1_score = calculate_measures(tn, tp, fn, fp)
    nb_accuracy_list.append(nb_accuracy)
    nb_precision_list.append(nb_precision)
    nb_recall_list.append(nb_recall)
    nb_sensitivity_list.append(nb_sensitivity)
    nb_specificity_list.append(nb_specificity)
    nb_false_positive_rate_list.append(nb_false_positive_rate)
    nb_false_negative_rate_list.append(nb_false_negative_rate)
    nb_misclassification_list.append(nb_misclassification)
    nb_f1_score_list.append(nb_f1_score)

    # ----- NEURAL NETWORK ----- #
    nn_model.fit(X_train_list[i], Y_train_list[i])
    nn_predictions = nn_model.predict(X_test_list[i])
    tn, fp, fn, tp = confusion_matrix(y_true=Y_test_list[i], y_pred=nn_predictions).ravel()
    nn_accuracy, nn_precision, nn_recall, nn_sensitivity, nn_specificity, nn_false_positive_rate, nn_false_negative_rate, nn_misclassification, nn_f1_score = calculate_measures(tn, tp, fn, fp)
    nn_accuracy_list.append(nn_accuracy)
    nn_precision_list.append(nn_precision)
    nn_recall_list.append(nn_recall)
    nn_sensitivity_list.append(nn_sensitivity)
    nn_specificity_list.append(nn_specificity)
    nn_false_positive_rate_list.append(nn_false_positive_rate)
    nn_false_negative_rate_list.append(nn_false_negative_rate)
    nn_misclassification_list.append(nn_misclassification)
    nn_f1_score_list.append(nn_f1_score)

    # ----- K-NEIGHBOURS CLASSIFIER ----- #
    kn_model.fit(X_train_list[i], Y_train_list[i])
    kn_predictions = kn_model.predict(X_test_list[i])
    tn, fp, fn, tp = confusion_matrix(y_true=Y_test_list[i], y_pred=kn_predictions).ravel()
    kn_accuracy, kn_precision, kn_recall, kn_sensitivity, kn_specificity, kn_false_positive_rate, kn_false_negative_rate, kn_misclassification, kn_f1_score = calculate_measures(tn, tp, fn, fp)
    kn_accuracy_list.append(kn_accuracy)
    kn_precision_list.append(kn_precision)
    kn_recall_list.append(kn_recall)
    kn_sensitivity_list.append(kn_sensitivity)
    kn_specificity_list.append(kn_specificity)
    kn_false_positive_rate_list.append(kn_false_positive_rate)
    kn_false_negative_rate_list.append(kn_false_negative_rate)
    kn_misclassification_list.append(kn_misclassification)
    kn_f1_score_list.append(kn_f1_score)


RF_accuracy = sum(rf_accuracy_list) / len(rf_accuracy_list)
RF_precision = sum(rf_precision_list) / len(rf_precision_list)
RF_recall = sum(rf_recall_list) / len(rf_recall_list)
RF_sensitivity = sum(rf_sensitivity_list) / len(rf_sensitivity_list)
RF_specificity = sum(rf_specificity_list) / len(rf_specificity_list)
RF_false_positive_rate = sum(rf_false_positive_rate_list) / len(rf_false_positive_rate_list)
RF_false_negative_rate = sum(rf_false_negative_rate_list) / len(rf_false_negative_rate_list)
RF_misclassification = sum(rf_misclassification_list) / len(rf_misclassification_list)
RF_f1_score = sum(rf_f1_score_list) / len(rf_f1_score_list)

print("Random Forest accuracy ==> ", RF_accuracy)
print("Random Forest precision ==> ", RF_precision)
print("Random Forest recall ==> ", RF_recall)
print("Random Forest sensitivity ==> ", RF_sensitivity)
print("Random Forest specificity",RF_specificity)
print("Random Forest false_positive_rate",RF_false_positive_rate)
print("\n")


DT_accuracy = sum(dt_accuracy_list) / len(dt_accuracy_list)
DT_precision = sum(dt_precision_list) / len(dt_precision_list)
DT_recall = sum(dt_recall_list) / len(dt_recall_list)
DT_sensitivity = sum(dt_sensitivity_list) / len(dt_sensitivity_list)
DT_specificity = sum(dt_specificity_list) / len(dt_specificity_list)
DT_false_positive_rate = sum(dt_false_positive_rate_list) / len(dt_false_positive_rate_list)
DT_false_negative_rate = sum(dt_false_negative_rate_list) / len(dt_false_negative_rate_list)
DT_misclassification = sum(dt_misclassification_list) / len(dt_misclassification_list)
DT_f1_score = sum(dt_f1_score_list) / len(dt_f1_score_list)

print("Decision Tree accuracy ==> ", DT_accuracy)
print("Decision Tree precision ==> ", DT_precision)
print("Decision Tree recall ==> ", DT_recall)
print("Decision Tree sensitivity ==> ", DT_sensitivity)
print("Decision Tree specificity",DT_specificity)
print("Decision Tree false_positive_rate",DT_false_positive_rate)
print("\n")


AB_accuracy = sum(ab_accuracy_list) / len(ab_accuracy_list)
AB_precision = sum(ab_precision_list) / len(ab_precision_list)
AB_recall = sum(ab_recall_list) / len(ab_recall_list)
AB_sensitivity = sum(ab_sensitivity_list) / len(ab_sensitivity_list)
AB_specificity = sum(ab_specificity_list) / len(ab_specificity_list)
AB_false_positive_rate = sum(ab_false_positive_rate_list) / len(ab_false_positive_rate_list)
AB_false_negative_rate = sum(ab_false_negative_rate_list) / len(ab_false_negative_rate_list)
AB_misclassification = sum(ab_misclassification_list) / len(ab_misclassification_list)
AB_f1_score = sum(ab_f1_score_list) / len(ab_f1_score_list)

print("AdaBoost accuracy ==> ", AB_accuracy)
print("AdaBoost precision ==> ", AB_precision)
print("AdaBoost recall ==> ", AB_recall)
print("AdaBoost sensitivity ==> ",AB_sensitivity)
print("AdaBoost specificity",AB_specificity)
print("AdaBoost false_positive_rate",AB_false_positive_rate)
print("\n")


SVM_accuracy = sum(svm_accuracy_list) / len(svm_accuracy_list)
SVM_precision = sum(svm_precision_list) / len(svm_precision_list)
SVM_recall = sum(svm_recall_list) / len(svm_recall_list)
SVM_sensitivity = sum(svm_sensitivity_list) / len(svm_sensitivity_list)
SVM_specificity = sum(svm_specificity_list) / len(svm_specificity_list)
SVM_false_positive_rate = sum(svm_false_positive_rate_list) / len(svm_false_positive_rate_list)
SVM_false_negative_rate = sum(svm_false_negative_rate_list) / len(svm_false_negative_rate_list)
SVM_misclassification = sum(svm_misclassification_list) / len(svm_misclassification_list)
SVM_f1_score = sum(svm_f1_score_list) / len(svm_f1_score_list)

print("Support Vector Machine accuracy ==> ", SVM_accuracy)
print("Support Vector Machine precision ==> ", SVM_precision)
print("Support Vector Machine recall ==> ", SVM_recall)
print("Support Vector Machine sensitivity ==> ", SVM_sensitivity)
print("Support Vector Machine specificity",SVM_specificity)
print("Support Vector Machine false_positive_rate",SVM_false_positive_rate)
print("\n")


NB_accuracy = sum(nb_accuracy_list) / len(nb_accuracy_list)
NB_precision = sum(nb_precision_list) / len(nb_precision_list)
NB_recall = sum(nb_recall_list) / len(nb_recall_list)
NB_sensitivity = sum(nb_sensitivity_list) / len(nb_sensitivity_list)
NB_specificity = sum(nb_specificity_list) / len(nb_specificity_list)
NB_false_positive_rate = sum(nb_false_positive_rate_list) / len(nb_false_positive_rate_list)
NB_false_negative_rate = sum(nb_false_negative_rate_list) / len(nb_false_negative_rate_list)
NB_misclassification = sum(nb_misclassification_list) / len(nb_misclassification_list)
NB_f1_score = sum(nb_f1_score_list) / len(nb_f1_score_list)

print("Gaussian Naive Bayes accuracy ==> ", NB_accuracy)
print("Gaussian Naive Bayes precision ==> ", NB_precision)
print("Gaussian Naive Bayes recall ==> ", NB_recall)
print("Gaussian Naive Bayes sensitivity ==> ", NB_sensitivity)
print("Gaussian Naive Bayes specificity",NB_specificity)
print("Gaussian Naive Bayes false_positive_rate",NB_false_positive_rate)
print("\n")


NN_accuracy = sum(nn_accuracy_list) / len(nn_accuracy_list)
NN_precision = sum(nn_precision_list) / len(nn_precision_list)
NN_recall = sum(nn_recall_list) / len(nn_recall_list)
NN_sensitivity = sum(nn_sensitivity_list) / len(nn_sensitivity_list)
NN_specificity = sum(nn_specificity_list) / len(nn_specificity_list)
NN_false_positive_rate = sum(nn_false_positive_rate_list) / len(nn_false_positive_rate_list)
NN_false_negative_rate = sum(nn_false_negative_rate_list) / len(nn_false_negative_rate_list)
NN_misclassification = sum(nn_misclassification_list) / len(nn_misclassification_list)
NN_f1_score = sum(nn_f1_score_list) / len(nn_f1_score_list)

print("Neural Network accuracy ==> ", NN_accuracy)
print("Neural Network precision ==> ", NN_precision)
print("Neural Network recall ==> ", NN_recall)
print("Neural Network sensitivity ==> ", NN_sensitivity)
print("Neural Network specificity",NN_specificity)
print("Neural Network false_positive_rate",NN_false_positive_rate)
print("\n")


KN_accuracy = sum(kn_accuracy_list) / len(kn_accuracy_list)
KN_precision = sum(kn_precision_list) / len(kn_precision_list)
KN_recall = sum(kn_recall_list) / len(kn_recall_list)
KN_sensitivity = sum(kn_sensitivity_list) / len(kn_sensitivity_list)
KN_specificity = sum(kn_specificity_list) / len(kn_specificity_list)
KN_false_positive_rate = sum(kn_false_positive_rate_list) / len(kn_false_positive_rate_list)
KN_false_negative_rate = sum(kn_false_negative_rate_list) / len(kn_false_negative_rate_list)
KN_misclassification = sum(kn_misclassification_list) / len(kn_misclassification_list)
KN_f1_score = sum(kn_f1_score_list) / len(kn_f1_score_list)

print("K-Neighbours Classifier accuracy ==> ", KN_accuracy)
print("K-Neighbours Classifier precision ==> ", KN_precision)
print("K-Neighbours Classifier recall ==> ", KN_recall)
print("K-Neighbours Classifier sensitivity ==> ", KN_sensitivity)
print("K-Neighbours Classifier specificity",KN_specificity)
print("K-Neighbours Classifier false_positive_rate",KN_false_positive_rate)
print("\n")


data = {'accuracy': [NB_accuracy, SVM_accuracy, DT_accuracy, RF_accuracy, AB_accuracy, NN_accuracy, KN_accuracy],
        'precision': [NB_precision, SVM_precision, DT_precision, RF_precision, AB_precision, NN_precision, KN_precision],
        'recall': [NB_recall, SVM_recall, DT_recall, RF_recall, AB_recall, NN_recall, KN_recall],
        'sensitivity' : [NB_sensitivity, SVM_sensitivity, DT_sensitivity, RF_sensitivity, AB_sensitivity, NN_sensitivity, KN_sensitivity],
        'specificity' : [NB_specificity, SVM_specificity, DT_specificity, RF_specificity, AB_specificity, NN_specificity, KN_specificity],
        'misclassification' : [NB_misclassification, SVM_misclassification, DT_misclassification, RF_misclassification, AB_misclassification, NN_misclassification, KN_misclassification],
        'f1 score' : [NB_f1_score, SVM_f1_score, DT_f1_score, RF_f1_score, AB_f1_score, NN_f1_score, KN_f1_score],
        'false positive rate' : [NB_false_positive_rate, SVM_false_positive_rate, DT_false_positive_rate, RF_false_positive_rate, AB_false_positive_rate, NN_false_positive_rate, KN_false_positive_rate],
        'false negative rate' : [NB_false_negative_rate, SVM_false_negative_rate, DT_false_negative_rate, RF_false_negative_rate, AB_false_negative_rate, NN_false_negative_rate, KN_false_negative_rate]
        }

index = ['NB', 'SVM', 'DT', 'RF', 'AB', 'NN', 'KN']

df_results = pd.DataFrame(data=data, index=index)

print(df_results)

# visualize the dataframe
ax = df_results.plot.bar(rot=0)
plt.show()