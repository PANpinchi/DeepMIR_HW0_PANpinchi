import os
import numpy as np
import argparse
import matplotlib.pyplot as plt

from sklearn.utils.multiclass import unique_labels
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, top_k_accuracy_score, confusion_matrix
import joblib


def load_preprocessed_data(input_dir):
    X_test = np.load(os.path.join(input_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(input_dir, 'y_test.npy'))
    return X_test, y_test


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          top_1=0.,
                          top_3=0.,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label\n(Top 1: {:.4f}, Top 3: {:.4f})'.format(top_1, top_3))

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    # fig.tight_layout()
    return ax


def main(model_type='knn', plot_cm=False):
    # Dataset and DataLoader initialization
    input_dir = './datasets'
    X_test, y_test = load_preprocessed_data(input_dir)

    # Ensure that we know all possible classes
    unique_labels = np.arange(11)

    # ML Model
    if model_type == 'knn':
        model_name = 'k-NN'
        model = KNeighborsClassifier(n_neighbors=5)
    elif model_type == 'rf':
        model_name = 'Random Forest'
        model = RandomForestClassifier(n_estimators=100)
    elif model_type == 'dt':
        model_name = 'Decision Tree'
        model = DecisionTreeClassifier()
    elif model_type == 'svm':
        model_name = 'SVM'
        model = SVC(probability=True)  # Enable probability estimates for top-k accuracy
    elif model_type == 'lr':
        model_name = 'Logistic Regression'
        model = LogisticRegression(max_iter=200)  # Increase max_iter if convergence issues arise
    else:
        raise ValueError("Unsupported model type: {}".format(model_type))

    # Load the saved model
    output_dir = './results_ML_model'
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, '{}_model.pkl'.format(model_name))
    model = joblib.load(model_path)
    print(f'Model loaded from {model_path}')

    # Predictions
    y_pred = model.predict(X_test)
    y_pred_probs = model.predict_proba(X_test)

    # Calculate Top-1 and Top-3 accuracy
    top_1_accuracy = accuracy_score(y_test, y_pred)
    top_3_accuracy = top_k_accuracy_score(y_test, y_pred_probs, k=3, labels=unique_labels)

    # Display results
    print('{} Test Accuracy:'.format(model_name))
    print('Top-1 Accuracy: {:.4f}'.format(top_1_accuracy))
    print('Top-3 Accuracy: {:.4f}'.format(top_3_accuracy))

    # Save the confusion matrix plot
    if plot_cm:
        class_names = np.array(['bass', 'brass', 'flute', 'guitar',
                                'keyboard', 'mallet', 'organ', 'reed',
                                'string', 'synth_lead', 'vocal'])
        title_content = 'Confusion Matrix ({})'.format(model_name)

        plot_confusion_matrix(y_test, y_pred, classes=class_names, normalize=True,
                              title=title_content, top_1=top_1_accuracy, top_3=top_3_accuracy)
        plt.savefig('results_ML_model/{}_confusion_matrix.png'.format(model_name))
        print(f'Confusion matrix saved at: ', '{}/{}_confusion_matrix.png'.format(output_dir, model_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a machine learning model on Nsynth dataset.")
    parser.add_argument('--model_type', type=str, default='knn', choices=['knn', 'rf', 'dt', 'svm', 'lr'],
                        help="The type of model to train: 'knn' for k-Nearest Neighbors, 'rf' for Random Forest, 'dt' for Decision Tree, 'svm' for Support Vector Machine, 'lr' for Logistic Regression")
    parser.add_argument('--plot_confusion_matrix', action='store_true', default=False,
                        help='Whether to plot confusion matrix.')

    args = parser.parse_args()

    main(model_type=args.model_type, plot_cm=args.plot_confusion_matrix)
