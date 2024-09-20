import os
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, top_k_accuracy_score, confusion_matrix
from tqdm import tqdm

from dataset import NsynthDataset
from model import ShortChunkCNN


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


def test_model(model, test_loader, device, use_log, plot_cm):
    # After training, run the test phase
    all_preds = []
    all_preds_probs = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(tqdm(test_loader, desc="Testing")):
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)

            # Track loss and predictions
            all_preds.append(outputs.argmax(dim=1).cpu().numpy())
            all_preds_probs.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_preds_probs = np.concatenate(all_preds_probs)
        all_labels = np.concatenate(all_labels)
        top_1_accuracy = accuracy_score(all_labels, all_preds)
        unique_labels = np.arange(11)  # Assuming 11 classes
        top_3_accuracy = top_k_accuracy_score(all_labels, all_preds_probs, k=3, labels=unique_labels)

    # Display results
    if use_log:
        print('Test Accuracy (with taking the log):')
    else:
        print('Test Accuracy (without taking the log):')
    print('Top-1 Accuracy: {:.4f}'.format(top_1_accuracy))
    print('Top-3 Accuracy: {:.4f}'.format(top_3_accuracy))

    # Plot confusion matrix
    if plot_cm:
        class_names = np.array(['bass', 'brass', 'flute', 'guitar',
                                'keyboard', 'mallet', 'organ', 'reed',
                                'string', 'synth_lead', 'vocal'])
        # Create the results directory if it doesn't exist
        output_dir = './results_DL_model'
        os.makedirs(output_dir, exist_ok=True)
        if use_log:
            name = 'with_log'
            title_content = 'Confusion Matrix (with taking the log)'
        else:
            name = 'without_log'
            title_content = 'Confusion Matrix (without taking the log)'
        plot_confusion_matrix(all_labels, all_preds, classes=class_names, normalize=True,
                              title=title_content, top_1=top_1_accuracy, top_3=top_3_accuracy)
        plt.savefig('{}/{}_confusion_matrix.png'.format(output_dir, name))
        print('Confusion matrix saved at: ', '{}/{}_confusion_matrix.png'.format(output_dir, name))


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train ShortChunkCNN model.')
    parser.add_argument('--use_log', action='store_true', default=False,
                        help='Whether to use logarithmic features.')
    parser.add_argument('--plot_confusion_matrix', action='store_true', default=False,
                        help='Whether to plot confusion matrix.')
    args = parser.parse_args()

    print('=== use_log ===> ', args.use_log)

    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device: {}'.format(device))

    # Model initialization
    model = ShortChunkCNN(n_channels=128,
                          sample_rate=16000,
                          n_fft=1004,
                          f_min=0.0,
                          f_max=8000.0,
                          n_mels=128,
                          n_class=11,
                          use_log=args.use_log)
    if args.use_log:
        model_path = './results_DL_model/best_model_with_log.pth'
    else:
        model_path = './results_DL_model/best_model_without_log.pth'
    model.load_state_dict(torch.load(model_path))
    print(f'Model weights loaded from {model_path}')
    model = model.to(device)

    # Dataset and DataLoader initialization
    test_audio_dir = './datasets/nsynth-test/audio'
    test_json_file = './datasets/nsynth-test/examples.json'

    nsynth_test_dataset = NsynthDataset(test_audio_dir, test_json_file)

    # Example usage
    test_loader = DataLoader(nsynth_test_dataset, batch_size=32, shuffle=False, num_workers=4)

    test_model(model, test_loader, device, use_log=args.use_log, plot_cm=args.plot_confusion_matrix)


if __name__ == '__main__':
    main()
