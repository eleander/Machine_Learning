# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# # Classification 
# 
# The most common supervised learning tasks are regression (predicting values) and classification (predicting classes)
# %% [markdown]
# ## Setup

# %%

# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

# Common imports
import numpy as np
import os

# to make this notebook's output stable across runs
np.random.seed(42)

# To plot pretty figures
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "classification"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

# %% [markdown]
# # MNIST
# 
# "In this chapter, we will be using the MNIST dataset, which is a set of 70,000 small images of digits handwritten by high school students and employees of the US Census Bureau. Each image is labeled with the digit it represents. This set has been studied so much that it is often called the “Hello World” of Machine Learning: whenever people come up with a new classification algorithm, they are curious to see how it will perform on MNIST. Whenever someone learns Machine Learning, sooner or later they tackle MNIST."
# 
# Warning: since Scikit-Learn 0.24, fetch_openml() returns a Pandas DataFrame by default. To avoid this and keep the same code as in the book, we use as_frame=False.

# %%
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
mnist.keys()

# %% [markdown]
# Datasets loaded by Scikit-Learn generally have a similar dictionary structure including:  
# - A DESCR key describing the dataset
# - A data key containing an array with one row per instance and one column per
# feature
# - A target key containing an array with the labels

# %%

X, y = mnist["data"], mnist["target"]
X.shape


# %%
28 * 28


# %%
y.shape

# %% [markdown]
# There are 70,000 images, and each image has 784 features. This is because each image
# is 28×28 pixels, and each feature simply represents one pixel’s intensity, from 0
# (white) to 255 (black). Let’s take a peek at one digit from the dataset. All you need to
# do is grab an instance’s feature vector, reshape it to a 28×28 array, and display it using
# Matplotlib’s imshow() function:

# %%
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt

some_digit = X[0]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap=mpl.cm.binary)
plt.axis("off")

save_fig("some_digit_plot")
plt.show()


# %%
y[0]


# %%
y = y.astype(np.uint8)


# %%
def plot_digit(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap = mpl.cm.binary,
               interpolation="nearest")
    plt.axis("off")


# %%
# EXTRA
def plot_digits(instances, images_per_row=10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size,size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap = mpl.cm.binary, **options)
    plt.axis("off")


# %%
plt.figure(figsize=(9,9))
example_images = X[:100]
plot_digits(example_images, images_per_row=10)
save_fig("more_digits_plot")
plt.show()

# %% [markdown]
# Creating a test set! The MNIST data set is already split up. The first 60,000 items are the training set. The last 10,000 items are the test set

# %%
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# %% [markdown]
# "Let’s also shuffle the training set; this will guarantee that all cross-validation folds will be similar (you don’t want one fold to be missing some digits). Moreover, some learning algorithms are sensitive to the order of the training instances, and they perform poorly if they get many similar instances in a row. Shuffling the dataset ensures that this won’t happen"

# %%
import numpy as np
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

# %% [markdown]
# # Binary classifier
# %% [markdown]
# The 5 classifier. Can determine if a picture is 5 or is not 5

# %%
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

# %% [markdown]
# Stochastic Gradient Descent is a good classifier to start with! It can handle very large datasets efficiently. This is because SGD deals with training instances independently, one at a time (which also makes SGD well suited for online learning)  
# 
# Note: The SGDClassifier relies on randomness during training (hence
# the name “stochastic”). If you want reproducible results, you
# should set the random_state parameter

# %%
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
sgd_clf.fit(X_train[:1000], y_train_5[:1000])

# %% [markdown]
# Now it can detect images for 5! some_digit is 5 so it guessed correctly

# %%
sgd_clf.predict([some_digit])

# %% [markdown]
# ## Performance Measures
# 
# "Evaluating a classifier is often significantly trickier than evaluating a regressor, so we will spend a large part of this chapter on this topic."
# %% [markdown]
# ## Measuring Accuracy Using Cross-Validation
# %% [markdown]
# "The StratifiedKFold class performs stratified sampling (as explained in Chapter 2) to produce folds that contain a representative ratio of each class. At each iteration the code creates a clone of the classifier, trains that clone on the training folds, and makes predictions on the test fold. Then it counts the number of correct predictions and outputs the ratio of correct predictions."

# %%
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

skfolds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = y_train_5[train_index]
    X_test_fold = X_train[test_index]
    y_test_fold = y_train_5[test_index]

    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct / len(y_pred))

# %% [markdown]
# "Let’s use the cross_val_score() function to evaluate your SGDClassifier model using K-fold cross-validation, with three folds. Remember that K-fold crossvalidation means splitting the training set into K-folds (in this case, three), then making predictions and evaluating them on each fold using a model trained on the remaining folds (see Chapter 2):"

# %%
from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")

# %% [markdown]
# Above 95% accuracy!!!!
# %% [markdown]
# Wait..." let’s look at a very dumb classifier that just classifies every single image in the “not-5” class:"

# %%

from sklearn.base import BaseEstimator
class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)


# %%
never_5_clf = Never5Classifier()
cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy")

# %% [markdown]
# "That’s right, it has over 90% accuracy! This is simply because only about 10% of the
# images are 5s, so if you always guess that an image is not a 5, you will be right about
# 90% of the time. Beats Nostradamus.  
# 
# This demonstrates why accuracy is generally not the preferred performance measure
# for classifiers, especially when you are dealing with skewed datasets (i.e., when some
# classes are much more frequent than others)."
# %% [markdown]
# # Confusion Matrix
# 
# "A much better way to evaluate the performance of a classifier is to look at the confusion matrix. The general idea is to count the number of times instances of class A are classified as class B. For example, to know the number of times the classifier confused images of 5s with 3s, you would look in the 5th row and 3rd column of the confusion matrix."  
# 
# "To compute the confusion matrix, you first need to have a set of predictions, so they can be compared to the actual targets. You could make predictions on the test set, but let’s keep it untouched for now (remember that you want to use the test set only at the very end of your project, once you have a classifier that you are ready to launch). Instead, you can use the cross_val_predict() function:"

# %%
from sklearn.model_selection import cross_val_predict

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

# %% [markdown]
# "Just like the cross_val_score() function, cross_val_predict() performs K-fold cross-validation, but instead of returning the evaluation scores, it returns the predictions made on each test fold. This means that you get a clean prediction for each instance in the training set (“clean” meaning that the prediction is made by a model that never saw the data during training"  
# 
# "Now you are ready to get the confusion matrix using the confusion_matrix() function.
# Just pass it the target classes (y_train_5) and the predicted classes
# (y_train_pred):"

# %%
from sklearn.metrics import confusion_matrix

confusion_matrix(y_train_5, y_train_pred)

# %% [markdown]
# Row = actual class  
# 
# Column = Predicted class
# 
# Row 1 = non-5 images  
# 53124 correctly classified as non-5, 1455 incorrectly identified (false positive)
#   
# Row 2 = 5 images  
# 949 wrongly classified as non-5 (false negatives), 4472 correctly classified as 5s (true positives)
# 
# 
# %% [markdown]
# "A perfect classifier would have only true positives and true
# negatives, so its confusion matrix would have nonzero values only on its main diagonal
# (top left to bottom right):"

# %%
y_train_perfect_predictions = y_train_5  # pretend we reached perfection
confusion_matrix(y_train_5, y_train_perfect_predictions)

# %% [markdown]
# "The confusion matrix gives you a lot of information, but sometimes you may prefer a
# more concise metric. An interesting one to look at is the accuracy of the positive predictions;
# this is called the precision of the classifier (Equation 3-1)."
# 
# "TP is the number of true positives, and FP is the number of false positives.
# A trivial way to have perfect precision is to make one single positive prediction and
# ensure it is correct (precision = 1/1 = 100%). This would not be very useful since the
# classifier would ignore all but one positive instance. So precision is typically used
# along with another metric named recall, also called sensitivity or true positive rate
# (TPR): this is the ratio of positive instances that are correctly detected by the classifier
# (Equation 3-2)."   
# 
# "An interesting one to look at is the accuracy of the positive predictions;
# this is called the precision of the classifier"
# 
# "So precision is typically used
# along with another metric named recall, also called sensitivity or true positive rate
# (TPR): this is the ratio of positive instances that are correctly detected by the classifier"
# 
# Pg 85
# %% [markdown]
# Scikit learn provides a method to calculate the precision score

# %%
from sklearn.metrics import precision_score, recall_score

precision_score(y_train_5, y_train_pred)


# %%
cm = confusion_matrix(y_train_5, y_train_pred)
cm[1, 1] / (cm[0, 1] + cm[1, 1])

# %% [markdown]
# Scikit learn provides a method to calculate the recall score

# %%
recall_score(y_train_5, y_train_pred)


# %%
cm[1, 1] / (cm[1, 0] + cm[1, 1])

# %% [markdown]
# "Now your 5-detector does not look as shiny as it did when you looked at its accuracy.
# When it claims an image represents a 5, it is correct only 75% of the time. Moreover,
# it only detects 82% of the 5s."
# %% [markdown]
# Scikit learn provides a method to determine the harmonic mean of precision and recall

# %%
from sklearn.metrics import f1_score

f1_score(y_train_5, y_train_pred)


# %%
cm[1, 1] / (cm[1, 1] + (cm[1, 0] + cm[0, 1]) / 2)

# %% [markdown]
# 
# %% [markdown]
# ## Precision/Recall Tradeoff
# 
# "The F1 score favors classifiers that have similar precision and recall. This is not always what you want: in some contexts you mostly care about precision, and in other contexts you really care about recall. For example, if you trained a classifier to detect videos
# that are safe for kids, you would probably prefer a classifier that rejects many
# good videos (low recall) but keeps only safe ones (high precision), rather than a classifier
# that has a much higher recall but lets a few really bad videos show up in your
# product (in such cases, you may even want to add a human pipeline to check the classifier’s
# video selection). On the other hand, suppose you train a classifier to detect
# shoplifters on surveillance images: it is probably fine if your classifier has only 30%
# precision as long as it has 99% recall (sure, the security guards will get a few false
# alerts, but almost all shoplifters will get caught).
# Unfortunately, you can’t have it both ways: increasing precision reduces recall, and
# vice versa. This is called the precision/recall tradeoff."   
# 
# 
# pg 87

# %%
y_scores = sgd_clf.decision_function([some_digit])
y_scores


# %%
threshold = 0
y_some_digit_pred = (y_scores > threshold)


# %%
y_some_digit_pred


# %%
threshold = 8000
y_some_digit_pred = (y_scores > threshold)
y_some_digit_pred


# %%
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,
                             method="decision_function")


# %%

from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)


# %%
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.legend(loc="center right", fontsize=16) # Not shown in the book
    plt.xlabel("Threshold", fontsize=16)        # Not shown
    plt.grid(True)                              # Not shown
    plt.axis([-50000, 50000, 0, 1])             # Not shown



recall_90_precision = recalls[np.argmax(precisions >= 0.90)]
threshold_90_precision = thresholds[np.argmax(precisions >= 0.90)]


plt.figure(figsize=(8, 4))                                                                  # Not shown
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.plot([threshold_90_precision, threshold_90_precision], [0., 0.9], "r:")                 # Not shown
plt.plot([-50000, threshold_90_precision], [0.9, 0.9], "r:")                                # Not shown
plt.plot([-50000, threshold_90_precision], [recall_90_precision, recall_90_precision], "r:")# Not shown
plt.plot([threshold_90_precision], [0.9], "ro")                                             # Not shown
plt.plot([threshold_90_precision], [recall_90_precision], "ro")                             # Not shown
save_fig("precision_recall_vs_threshold_plot")                                              # Not shown
plt.show()


# %%
(y_train_pred == (y_scores > 0)).all()


# %%
def plot_precision_vs_recall(precisions, recalls):
    plt.plot(recalls, precisions, "b-", linewidth=2)
    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    plt.axis([0, 1, 0, 1])
    plt.grid(True)

plt.figure(figsize=(8, 6))
plot_precision_vs_recall(precisions, recalls)
plt.plot([recall_90_precision, recall_90_precision], [0., 0.9], "r:")
plt.plot([0.0, recall_90_precision], [0.9, 0.9], "r:")
plt.plot([recall_90_precision], [0.9], "ro")
save_fig("precision_vs_recall_plot")
plt.show()


# %%
threshold_90_precision = thresholds[np.argmax(precisions >= 0.90)]


# %%
threshold_90_precision


# %%
y_train_pred_90 = (y_scores >= threshold_90_precision)


# %%
precision_score(y_train_5, y_train_pred_90)


# %%
recall_score(y_train_5, y_train_pred_90)

# %% [markdown]
# "Great, you have a 90% precision classifier (or close enough)! As you can see, it is
# fairly easy to create a classifier with virtually any precision you want: just set a high
# enough threshold, and you’re done. Hmm, not so fast. A high-precision classifier is
# not very useful if its recall is too low!"
# %% [markdown]
# ## The Roc Curve 
# 
# "The receiver operating characteristic (ROC) curve is another common tool used with
# binary classifiers. It is very similar to the precision/recall curve, but instead of plotting
# precision versus recall, the ROC curve plots the true positive rate (another name
# for recall) against the false positive rate. The FPR is the ratio of negative instances that
# are incorrectly classified as positive. It is equal to one minus the true negative rate,
# which is the ratio of negative instances that are correctly classified as negative. The
# TNR is also called specificity. Hence the ROC curve plots sensitivity (recall) versus
# 1 – specificity."

# %%
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)


# %%
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')                           # dashed diagonal
    plt.axis([0, 1, 0, 1])                                    # Not shown in the book
    plt.xlabel('False Positive Rate (Fall-Out)', fontsize=16) # Not shown
    plt.ylabel('True Positive Rate (Recall)', fontsize=16)    # Not shown
    plt.grid(True)                                            # Not shown

plt.figure(figsize=(8, 6))                                    # Not shown
plot_roc_curve(fpr, tpr)
fpr_90 = fpr[np.argmax(tpr >= recall_90_precision)]           # Not shown
plt.plot([fpr_90, fpr_90], [0., recall_90_precision], "r:")   # Not shown
plt.plot([0.0, fpr_90], [recall_90_precision, recall_90_precision], "r:")  # Not shown
plt.plot([fpr_90], [recall_90_precision], "ro")               # Not shown
save_fig("roc_curve_plot")                                    # Not shown
plt.show()

# %% [markdown]
# "Once again there is a tradeoff: the higher the recall (TPR), the more false positives
# (FPR) the classifier produces. The dotted line represents the ROC curve of a purely
# random classifier; a good classifier stays as far away from that line as possible (toward
# the top-left corner)."   
# 
# "One way to compare classifiers is to measure the area under the curve (AUC). A perfect
# classifier will have a ROC AUC equal to 1, whereas a purely random classifier will
# have a ROC AUC equal to 0.5. Scikit-Learn provides a function to compute the ROC
# AUC:"

# %%
from sklearn.metrics import roc_auc_score

roc_auc_score(y_train_5, y_scores)

# %% [markdown]
# Important!!!   
# 
# "Since the ROC curve is so similar to the precision/recall (or PR)
# curve, you may wonder how to decide which one to use. As a rule
# of thumb, you should prefer the PR curve whenever the positive
# class is rare or when you care more about the false positives than
# the false negatives, and the ROC curve otherwise. For example,
# looking at the previous ROC curve (and the ROC AUC score), you
# may think that the classifier is really good. But this is mostly
# because there are few positives (5s) compared to the negatives
# (non-5s). In contrast, the PR curve makes it clear that the classifier
# has room for improvement (the curve could be closer to the topright
# corner)."
# %% [markdown]
# "Let’s train a RandomForestClassifier and compare its ROC curve and ROC AUC
# score to the SGDClassifier. First, you need to get scores for each instance in the
# training set. But due to the way it works (see Chapter 7), the RandomForestClassi
# fier class does not have a decision_function() method. Instead it has a pre
# dict_proba() method. Scikit-Learn classifiers generally have one or the other. The
# predict_proba() method returns an array containing a row per instance and a column
# per class, each containing the probability that the given instance belongs to the
# given class (e.g., 70% chance that the image represents a 5):"

# %%
from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3,
                                    method="predict_proba")

# %% [markdown]
# "But to plot a ROC curve, you need scores, not probabilities. A simple solution is to
# use the positive class’s probability as the score"

# %%
y_scores_forest = y_probas_forest[:, 1] # score = proba of positive class
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5,y_scores_forest)


# %%
recall_for_forest = tpr_forest[np.argmax(fpr_forest >= fpr_90)]

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, "b:", linewidth=2, label="SGD")
plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
plt.plot([fpr_90, fpr_90], [0., recall_90_precision], "r:")
plt.plot([0.0, fpr_90], [recall_90_precision, recall_90_precision], "r:")
plt.plot([fpr_90], [recall_90_precision], "ro")
plt.plot([fpr_90, fpr_90], [0., recall_for_forest], "r:")
plt.plot([fpr_90], [recall_for_forest], "ro")
plt.grid(True)
plt.legend(loc="lower right", fontsize=16)
save_fig("roc_curve_comparison_plot")
plt.show()

# %% [markdown]
# "As you can see in Figure 3-7, the RandomForestClassifier’s ROC curve looks much
# better than the SGDClassifier’s: it comes much closer to the top-left corner. As a
# result, its ROC AUC score is also significantly better:"

# %%
roc_auc_score(y_train_5, y_scores_forest)

# %% [markdown]
# Good precision and recall too!!!

# %%
y_train_pred_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3)
precision_score(y_train_5, y_train_pred_forest)


# %%
recall_score(y_train_5, y_train_pred_forest)

# %% [markdown]
# # Multiclass Classification
# 
# "Whereas binary classifiers distinguish between two classes, multiclass classifiers (also called multinomial classifiers) can distinguish between more than two classes."
# 
# "Some algorithms (such as Random Forest classifiers or naive Bayes classifiers) are capable of handling multiple classes directly. Others (such as Support Vector Machine classifiers or Linear classifiers) are strictly binary classifiers. However, there are various strategies that you can use to perform multiclass classification using multiple binary classifiers."
# 
# "For example, one way to create a system that can classify the digit images into 10 classes (from 0 to 9) is to train 10 binary classifiers, one for each digit (a 0-detector, a 1-detector, a 2-detector, and so on). Then when you want to classify an image, you get the decision score from each classifier for that image and you select the class whose classifier outputs the highest score. This is called the one-versus-all (OvA) strategy (also called one-versus-the-rest)."
# 
# "Another strategy is to train a binary classifier for every pair of digits: one to distinguish 0s and 1s, another to distinguish 0s and 2s, another for 1s and 2s, and so on. This is called the one-versus-one (OvO) strategy. If there are N classes, you need to train N × (N – 1) / 2 classifiers. For the MNIST problem, this means training 45 binary classifiers! When you want to classify an image, you have to run the image through all 45 classifiers and see which class wins the most duels. The main advantage of OvO is that each classifier only needs to be trained on the part of the training set for the two classes that it must distinguish."
# 
# "Some algorithms (such as Support Vector Machine classifiers) scale poorly with the
# size of the training set, so for these algorithms OvO is preferred since it is faster to
# train many classifiers on small training sets than training few classifiers on large
# training sets. For most binary classification algorithms, however, OvA is preferred."
# %% [markdown]
# "Scikit-Learn detects when you try to use a binary classification algorithm for a multiclass
# classification task, and it automatically runs OvA (except for SVM classifiers for
# which it uses OvO). Let’s try this with the SGDClassifier:"

# %%
sgd_clf.fit(X_train[:1000], y_train[:1000])
sgd_clf.predict([some_digit])

# %% [markdown]
# "That was easy! This code trains the SGDClassifier on the training set using the original target classes from 0 to 9 (y_train), instead of the 5-versus-all target classes (y_train_5). Then it makes a prediction (a correct one in this case). Under the hood, Scikit-Learn actually trained 10 binary classifiers, got their decision scores for the image, and selected the class with the highest score."

# %%
sgd_clf.decision_function([some_digit])

# %% [markdown]
# Now let's try with a Support Vector Machine
# %% [markdown]
# 

# %%
from sklearn.svm import SVC

svm_clf = SVC(gamma="auto", random_state=42)
svm_clf.fit(X_train[:1000], y_train[:1000]) # y_train, not y_train_5
svm_clf.predict([some_digit])


# %%
some_digit_scores = svm_clf.decision_function([some_digit])
some_digit_scores


# %%
np.argmax(some_digit_scores)


# %%
svm_clf.classes_


# %%
svm_clf.classes_[5]

# %% [markdown]
# You can force Scikit Learn to use OVA (OneVsRestClassifier) or OVO (OneVsOneClassifier)

# %%
from sklearn.multiclass import OneVsRestClassifier
ovr_clf = OneVsRestClassifier(SVC(gamma="auto", random_state=42))
ovr_clf.fit(X_train[:1000], y_train[:1000])
ovr_clf.predict([some_digit])


# %%
# from sklearn.multiclass import OneVsOneClassifier
# ovo_clf = OneVsOneClassifier(SGDClassifier(random_state=42))
# ovo_clf.fit(X_train, y_train)
# ovo_clf.predict([some_digit])


# %%
# len(ovo_clf.estimators_)

# %% [markdown]
# It is also easy to train a random forests. This time Scikit-Learn did not have to run OvA or OvO because Random Forest
# classifiers can directly classify instances into multiple classes. You can call
# predict_proba() to get the list of probabilities that the classifier assigned to each
# instance for each class

# %%
# forest_clf.fit(X_train, y_train)
# forest_clf.predict([some_digit])
# forest_clf.predict_proba([some_digit])


# %%
len(ovr_clf.estimators_)

# %% [markdown]
# Remember to cross validate!

# %%
cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")

# %% [markdown]
# "It gets over 84% on all test folds. If you used a random classifier, you would get 10%
# accuracy, so this is not such a bad score, but you can still do much better. For example,
# simply scaling the inputs (as discussed in Chapter 2) increases accuracy above
# 90%:"

# %%
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")

# %% [markdown]
# ## Error Analyis 
# 
# "Of course, if this were a real project, you would follow the steps in your Machine
# Learning project checklist (see Appendix B): exploring data preparation options, trying
# out multiple models, shortlisting the best ones and fine-tuning their hyperparameters
# using GridSearchCV, and automating as much as possible, as you did in the
# previous chapter. Here, we will assume that you have found a promising model and
# you want to find ways to improve it. One way to do this is to analyze the types of
# errors it makes."

# %%
y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
conf_mx = confusion_matrix(y_train, y_train_pred)
conf_mx


# %%



