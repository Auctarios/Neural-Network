from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve


def calculate_accuracy(y_pred, y):
    return np.mean([1 if i > 0.5 else 0 for i in y_pred] == y)

def calculate_precision(y_pred, y):
    y_pred = np.array([1 if i > 0.5 else 0 for i in y_pred])
    y = np.array([1 if i > 0.5 else 0 for i in y])
    TP = np.sum((y_pred == 1) & (y == 1))
    FP = np.sum((y_pred == 1) & (y == 0))
    return TP / (TP + FP)

def calculate_recall(y_pred, y):
    y_pred = np.array([1 if i > 0.5 else 0 for i in y_pred])
    y = np.array([1 if i > 0.5 else 0 for i in y])
    TP = np.sum((y_pred == 1) & (y == 1))
    FN = np.sum((y_pred == 0) & (y == 1))
    return TP / (TP + FN)

def calculate_f1_score(y_pred, y):
    precision = calculate_precision(y_pred, y)
    recall = calculate_recall(y_pred, y)
    return 2 * (precision * recall) / (precision + recall)

def plot_confusion_matrix(y_pred, y):
    cm = confusion_matrix(y, [1 if i > 0.5 else 0 for i in y_pred])
    labels = ['0', '1']
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()

def plot_roc_curve(y_pred, y):
    fpr1, tpr1, thresh1 = roc_curve(y, y_pred, pos_label=1)
    plt.figure(figsize=(4,3))
    plt.plot(fpr1, tpr1, linestyle='--',color='orange', label='Logistic Regression')

    plt.title('ROC curve')
    # x label
    plt.xlabel('False Positive Rate')
    # y label
    plt.ylabel('True Positive rate')
    plt.show()

def calculate_roc_auc(y_pred, y):
    return roc_auc_score(y, y_pred)

def eval_valid(y_pred, y):
    y = np.array(y)
    acc = calculate_accuracy(y_pred, y)
    prec = calculate_precision(y_pred, y)
    rec = calculate_recall(y_pred, y)
    f1_score = calculate_f1_score(y_pred, y)
    auc = calculate_roc_auc(y_pred, y)
    print(f"Accuracy: {acc}")
    print(f"Precision: {prec}")
    print(f"Recall: {rec}")
    print(f"F1-Score: {f1_score}")
    plot_confusion_matrix(y_pred, y)
    plot_roc_curve(y_pred, y)
    print(f"Area Under Curve (AUC) {auc}")
    return [acc, prec, rec, f1_score, auc]