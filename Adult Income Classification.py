from numpy import array, logspace, NaN
from pandas import concat, get_dummies, read_csv, set_option
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score, precision_recall_curve, precision_score, recall_score, roc_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils import resample
from skopt import BayesSearchCV
from skopt.space.space import Categorical, Integer 
from time import time
from matplotlib.pyplot import bar, box, figure, legend, plot, savefig, scatter, show, subplot, subplots, title, xlabel, ylabel
from warnings import filterwarnings
# Code for filtering out the warning.
filterwarnings("ignore")

# Defining whatever is needed.
df, Scales, Standardizer, PCA, CV = read_csv("Adult.csv", names = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"], header = None, index_col = None, delimiter = " *, *").copy(), {"age": "Standardization", "workclass": "One-hot", "education-num": "Standardization", "marital-status": "One-hot", "occupation": "One-hot", "relationship": "One-hot", "race": "One-hot", "sex": "One-hot", "hours-per-week": "Standardization", "native-country": "One-hot", "income": "One-hot"}, StandardScaler(), PCA(n_components = 20), StratifiedKFold(n_splits = 5, shuffle = True, random_state = 682)
# Classifiers with their raw model structure for being fine-tuned."
# Dictionary of Classifiers = {Known name of classifier: (Classifier(), BayesSearchCV(Target Parameters & ...))}
# 'linear' and 'poly' kernels  won't be checked for tuning SVM classification model because of magnitude of dataset.
Classifiers, ROCAUC, PCA_ROCAUC = {"SVM": (SVC, BayesSearchCV(estimator = SVC(), search_spaces = {"C": [0.1, 1, 10, 100], "gamma": [1, 0.1, 0.01], "kernel": ["linear", "poly", "rbf", "sigmoid"], "probability": [True]}, n_iter = 20, cv = CV)), "Naive Bayes": (GaussianNB, BayesSearchCV(estimator = GaussianNB(), search_spaces = {'priors': [None], "var_smoothing": logspace(0, -9, num = 100)}, scoring = "accuracy", cv = CV, n_jobs = 1, n_iter = 28, refit = False, random_state = 682)), "KNN": (KNeighborsClassifier, BayesSearchCV(estimator = KNeighborsClassifier(), search_spaces = {"algorithm": ["auto", "ball_tree", "kd_tree", "brute"], "n_neighbors": Integer(2, 40), 'p': Integer(1, 2), "weights": Categorical(["distance", "uniform"])}, scoring = "accuracy", cv = CV, n_jobs = 1, n_iter = 28, refit = False, random_state = 682)), "MLP": (MLPClassifier, BayesSearchCV(estimator = MLPClassifier(), search_spaces = {"activation": ["identity", "logistic", "relu", "tanh"], "alpha": [0.0001, 0.05], "early_stopping": [False], "learning_rate": ["adaptive", "constant", 'invscaling'], "max_iter": [100], "solver": ["adam", "sgd"], "warm_start": [False]}, scoring = "accuracy", cv = CV, n_jobs = 1, n_iter = 28, refit = False, random_state = 682))}, [], []

# All classifiers uses the same pattern for being tuned, trained and tested.
def Main(Classifier, bPCA = False):
    print(f"\n{Classifier} Modeling Started!")
    # Using Bayes Search approach for tuning model
    # Checking whether PCA approach is implied or not. 
    if bPCA:
        print("PCA transformation implied!")
        X_tr, X_v, X_tt = PCAX_train, PCAX_val, PCAX_test
    else:
        X_tr, X_v, X_tt = X_train, X_val, X_test

    CLF = Classifiers[Classifier]
    Model = CLF[1]
    startTime = time()
    # Obtaining best paramteres by using splitted validation data
    Model.fit(X_v, y_val)
    Duration = time() - startTime # Calculating how long fitting validation data takes
    print(f"\nModel's best score is:\n{Model.best_score_}\n\nModel's best parameters are:\n{Model.best_params_}")
    Model = CLF[0](**Model.best_params_) # = CLF[0](Optimal Parameters)
    print(f"\n{Model}")
    # Evaluation Time!
    Model.fit(X_tr, y_train)
    y_pred, y_pred_proba = Model.predict(X_tt), Model.predict_proba(X_tt)[:,1]
    if bPCA:
        PCA_ROCAUC.append(y_pred_proba)
    else:
        ROCAUC.append(y_pred_proba)

    print(f"\nModel Score: {Model.score(X_tt, y_pred)}\n\nAccuracy Score: {accuracy_score(y_test, y_pred)}\n\nPrecision Score: {precision_score(y_test, y_pred)}\n\nRecall Score: {recall_score(y_test, y_pred)}\n\nROC AUC Score: {roc_auc_score(y_test, y_pred_proba)}\n\nF1-Score: {f1_score(y_test, y_pred)}\n\nComputation Time: {Duration} UTC")

    # Comprehensive classification report.
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    ConfusionMatrixDisplay(confusion_matrix = confusion_matrix(y_test, y_pred), display_labels = ["Positive", "Negative"]).plot()

    # Plotting Precision-Recall Curve
    Precision, Recall, Threshold = precision_recall_curve(y_test, y_pred_proba)
    fig, ax = subplots(figsize = (6, 6))
    ax.plot(Recall, Precision, label = f"{Classifier} Classification", color = "firebrick")
    ax.set_title("Precision-Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    box(False)
    ax.legend()
    show()

# Showing all the columns in the table.
set_option("display.max_columns", None)
print(df.head())
print()
print(df.shape)
print()

# from above data we can say that education and flnwgt is not required to predict the income.
# education has one more representation in data by the variable "education_num"
# thus education can be removed.
# Features 'capital_gain' and 'capital_loss' are also useless.
df = df.drop(["fnlwgt", "education", "capital-gain", "capital-loss"], axis = 1)
print(df.head())
print(df.isnull().sum())
Features = df.columns
for Feature in Features:
    print(df[Feature].unique())

# missing values are in "?" form 
# thus we need to replace "?" with "NaN"
df = df.replace(["?"], NaN)
df.isnull().sum()

# Thus from above we can see that dataset has missing values.
# Now we have to replace these missing values with measure of central tendancy
# in this case we have to replace them with modes of the respective veriables.
for Feature in ["workclass", "occupation", "native-country"]:
    df[Feature].fillna(df[Feature].mode()[0], inplace = True)
print(df.isnull().sum())

# Balancing data
print(df["income"].value_counts())
# Downsampling records with incomes '<= 50k'
df = concat([resample(df[df["income"] == "<=50K"], replace = False, n_samples = 11687, random_state = 123), df[df["income"] == ">50K"]])
# Perfectly Balanced!
print(df["income"].value_counts())

# Preprocessing data records!
for Feature in Features:
    if Scales[Feature] == "Standardization":
        df[Feature] = Standardizer.fit_transform(array(df[Feature]).reshape(-1, 1))
    else:
        df = concat([df, get_dummies(df[[Feature]], dtype = int)], axis = 1).drop([Feature], axis = 1)
#income: 0 --> <=50k & 1 --> >50k
df.rename(columns = {"income_>50K": "income"}, inplace = True)
df = df.drop(["income_<=50K"], axis = 1)
# Thus from above we can see that we have removed all the missing values.
print(df.info())
print(df.describe)
"""
# Plotting encoded values of features
Features = df.columns.values
for Feature in Features:
  print(Feature)
  df[Feature].hist()
  show()
"""
# Creating X & Y and splitting the data into training and testing data set.
X, y = df.drop(["income"], axis = 1), df["income"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42) # 70:30
# 'test_size=0.5' split into 50% and 50%. The original data set is 30%; so, it will split into 15% equally.
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size = 0.5, random_state = 42) # 70:15:15
print(f"Training X shape: {X_train.shape}\nTraining Y shape: {y_train.shape}\nValidation X shape: {X_val.shape}\nValidation Y shape: {y_val.shape}\nTest X shape: {X_test.shape}\nTest Y shape: {y_test.shape}")
# Standardization once more!
X, X_train, X_val, X_test = Standardizer.fit_transform(X), Standardizer.fit_transform(X_train), Standardizer.fit_transform(X_val), Standardizer.transform(X_test)
#Applying the PCA. (For Feature Selection)
PCAX, PCAX_train, PCAX_val, PCAX_test = PCA.fit(X), PCA.fit_transform(X_train), PCA.transform(X_val), PCA.transform(X_test)

# Barplotting effect of PCA transformation (by variance)
figure(figsize = (25, 7)) 
subplot(1, 2, 1)
xlabel("PCA Feature")
ylabel("Variance")
title("PCA for Data Set")
bar(range(0, PCAX.explained_variance_ratio_.size), PCAX.explained_variance_ratio_)
show()
print(f"Explained variance ratio is:\n{PCA.explained_variance_ratio_}\n\n{PCA.n_components_}\n\nFeature vector of train set after applying PCA feature reduction method:\n{PCAX_train}\n\nFeature vector of test set after applying PCA feature reduction method:\n{PCAX_test}")

# Formatting
Colors, Targets, lw, Alpha = ["navy", "darkorange"], [0, 1], 2, 0.3
# 2 Components PCA
figure(2, figsize = (20, 8))
subplot(1, 2, 1)
PCAX = PCAX.transform(X)
for Color, i, Target in zip(Colors, [0, 1], Targets):
    scatter(PCAX[y == i, 0], PCAX[y == i, 1], color = Color, alpha = Alpha, lw = lw, label = Target)
legend(loc = "best", shadow = False, scatterpoints = 1)
title("First Two PCA Directions")
show()
# 3 Components PCA
ax = subplot(1, 2, 2, projection = "3d")
for Color, i, Target in zip(Colors, [0, 1], Targets):
    ax.scatter(PCAX[y == i, 0], PCAX[y == i, 1], PCAX[y == i, 2], color = Color, alpha = Alpha, lw = lw, label = Target)
legend(loc = "best", shadow = False, scatterpoints = 1)
ax.set_title("First Three PCA Directions")
ax.set_xlabel("1st Eigenvector")
ax.set_ylabel("2nd Eigenvector")
ax.set_zlabel("3rd Eigenvector")
# rotate the axes
ax.view_init(30, 10)
show()

# main()
for Classifier in Classifiers:
    Main(Classifier)
    Main(Classifier, True)
    
# Plotting ROC curves for all classifiers.
print("\nPlotting ROC curves for all classifiers without considering PCA transformations.")
# roc curve for tpr = fpr
FPR, TPR, _ = roc_curve(y_test, [0 for i in range(len(y_test))], pos_label = 1)
SVM_FPR, SVM_TPR, SVM_Threshold = roc_curve(y_test, ROCAUC[0], pos_label = 1)
NB_FPR, NB_TPR, NB_Threshold = roc_curve(y_test, PCA_ROCAUC[1], pos_label = 1)
KNN_FPR, KNN_TPR, KNN_Threshold = roc_curve(y_test, PCA_ROCAUC[2], pos_label = 1)
MLP_FPR, MLP_TPR, MLP_Threshold = roc_curve(y_test, PCA_ROCAUC[3], pos_label = 1)
plot(SVM_FPR, SVM_TPR, linestyle = "--", color = "black", label = "SVM")
plot(NB_FPR, NB_TPR, linestyle = "--", color = "orange", label = "Naive Bayes")
plot(KNN_FPR, KNN_TPR, linestyle = "--", color = "green", label = "KNN")
plot(MLP_FPR, MLP_TPR, linestyle = "--", color = "blue", label = "MLP")
plot(FPR, TPR, linestyle = "--", color = "red")
# title
title("ROC Curve")
# x label
xlabel("False Positive Rate")
# y label
ylabel("True Positive Rate")
legend(loc = "best")
savefig("ROC", dpi = 1200)
show()

print("\nPlotting ROC curves for all classifiers considering PCA transformations.")
PCA_SVM_FPR, PCA_SVM_TPR, PCA_SVM_Threshold = roc_curve(y_test, PCA_ROCAUC[0], pos_label = 1)
PCA_NB_FPR, PCA_NB_TPR, PCA_NB_Threshold = roc_curve(y_test, PCA_ROCAUC[1], pos_label = 1)
PCA_KNN_FPR, PCA_KNN_TPR, PCA_KNN_Threshold = roc_curve(y_test, PCA_ROCAUC[2], pos_label = 1)
PCA_MLP_FPR, PCA_MLP_TPR, PCA_MLP_Threshold = roc_curve(y_test, PCA_ROCAUC[3], pos_label = 1)
plot(PCA_SVM_FPR, PCA_SVM_TPR, linestyle = "--", color = "black", label = "SVM")
plot(PCA_NB_FPR, PCA_NB_TPR, linestyle = "--", color = "orange", label = "Naive Bayes")
plot(PCA_KNN_FPR, PCA_KNN_TPR, linestyle = "--", color = "green", label = "KNN")
plot(PCA_MLP_FPR, PCA_MLP_TPR, linestyle = "--", color = "blue", label = "MLP")
plot(FPR, TPR, linestyle = "--", color = "red")
# title
title("ROC Curve by PCA")
# x label
xlabel("False Positive Rate")
# y label
ylabel("True Positive Rate")
legend(loc = "best")
savefig("ROC", dpi = 1200)
show()

print("THE-END!!")
