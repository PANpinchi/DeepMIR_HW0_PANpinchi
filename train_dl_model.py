import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
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


def plot_loss_and_acc(num_epochs, train_losses, val_losses, train_accuracies, val_accuracies, use_log, output_dir):
    epochs = range(1, num_epochs + 1)

    plt.figure(figsize=(12, 5))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    if use_log:
        plt.title('Training and Validation Accuracy (with taking the log)')
    else:
        plt.title('Training and Validation Accuracy (without taking the log)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Save plots
    plt.tight_layout()
    if use_log:
        plt.savefig(f'{output_dir}/with_log_training_validation_curve.png')
        print('Training and validation curves saved at:', f'{output_dir}/with_log_training_validation_curve.png')
    else:
        plt.savefig(f'{output_dir}/without_log_training_validation_curve.png')
        print('Training and validation curves saved at:', f'{output_dir}/without_log_training_validation_curve.png')
    plt.show()


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for batch_idx, (inputs, labels) in enumerate(tqdm(train_loader)):
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Track loss and predictions
        running_loss += loss.item()
        all_preds.append(outputs.argmax(dim=1).cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    avg_loss = running_loss / len(train_loader)
    accuracy = accuracy_score(np.concatenate(all_labels), np.concatenate(all_preds))
    return avg_loss, accuracy


def validate_one_epoch(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(tqdm(val_loader)):
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Track loss and predictions
            running_loss += loss.item()
            all_preds.append(outputs.argmax(dim=1).cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    avg_loss = running_loss / len(val_loader)
    accuracy = accuracy_score(np.concatenate(all_labels), np.concatenate(all_preds))
    return avg_loss, accuracy


def test_one_epoch(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_preds_probs = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(tqdm(test_loader, desc="Testing")):
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Track loss and predictions
            running_loss += loss.item()
            all_preds.append(outputs.argmax(dim=1).cpu().numpy())
            all_preds_probs.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    avg_loss = running_loss / len(test_loader)
    all_preds = np.concatenate(all_preds)
    all_preds_probs = np.concatenate(all_preds_probs)
    all_labels = np.concatenate(all_labels)
    top_1_accuracy = accuracy_score(all_labels, all_preds)
    unique_labels = np.arange(11)  # Assuming 11 classes
    top_3_accuracy = top_k_accuracy_score(all_labels, all_preds_probs, k=3, labels=unique_labels)
    return avg_loss, all_labels, all_preds, top_1_accuracy, top_3_accuracy


# Training loop
def train_model(model, train_loader, val_loader, test_loader, criterion, optimizer, device, use_log, num_epochs=10):
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}:')
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)

        # Append losses and accuracies to lists
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}')

        # Save model if validation accuracy is improved
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if use_log:
                model_path = f'./results_DL_model/best_model_with_log.pth'
            else:
                model_path = f'./results_DL_model/best_model_without_log.pth'
            torch.save(model.state_dict(), model_path)
            print(f'Saved best model to {model_path}')

    # After training, run the test phase
    test_loss, test_labels, test_preds, top_1_accuracy, top_3_accuracy = test_one_epoch(model, test_loader, criterion, device)

    # Display results
    if use_log:
        print('Test Accuracy (with taking the log):')
    else:
        print('Test Accuracy (without taking the log):')
    print('Top-1 Accuracy: {:.4f}'.format(top_1_accuracy))
    print('Top-3 Accuracy: {:.4f}'.format(top_3_accuracy))

    # Plot confusion matrix
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
    plot_confusion_matrix(test_labels, test_preds, classes=class_names, normalize=True,
                          title=title_content, top_1=top_1_accuracy, top_3=top_3_accuracy)
    plt.savefig('{}/{}_confusion_matrix.png'.format(output_dir, name))
    print('Confusion matrix saved at: ', '{}/{}_confusion_matrix.png'.format(output_dir, name))

    # Plot training and validation losses and accuracies
    plot_loss_and_acc(num_epochs, train_losses, val_losses, train_accuracies, val_accuracies, use_log, output_dir)


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train ShortChunkCNN model.')
    parser.add_argument('--use_log', action='store_true', default=False, help='Whether to use logarithmic features.')
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
    model = model.to(device)

    # Dataset and DataLoader initialization
    subtrain_audio_dir = './datasets/nsynth-subtrain/audio'
    subtrain_json_file = './datasets/nsynth-subtrain/examples.json'
    valid_audio_dir = './datasets/nsynth-valid/audio'
    valid_json_file = './datasets/nsynth-valid/examples.json'
    test_audio_dir = './datasets/nsynth-test/audio'
    test_json_file = './datasets/nsynth-test/examples.json'

    nsynth_dataset = NsynthDataset(subtrain_audio_dir, subtrain_json_file)
    nsynth_val_dataset = NsynthDataset(valid_audio_dir, valid_json_file)
    nsynth_test_dataset = NsynthDataset(test_audio_dir, test_json_file)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Example usage
    train_loader = DataLoader(nsynth_dataset, batch_size=256, shuffle=True, num_workers=4)
    val_loader = DataLoader(nsynth_val_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(nsynth_test_dataset, batch_size=32, shuffle=False, num_workers=4)

    train_model(model, train_loader, val_loader, test_loader, criterion, optimizer, device, use_log=args.use_log, num_epochs=10)


if __name__ == '__main__':
    main()
