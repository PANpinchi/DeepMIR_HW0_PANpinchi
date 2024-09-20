import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_mel_spectrogram(y, sr, title, results_dir, results_dir_db):
    # Compute Mel-spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512)

    fig, ax = plt.subplots(figsize=(10, 7))

    # Display Mel-spectrogram
    img = librosa.display.specshow(mel_spectrogram, x_axis='time', y_axis='mel', sr=sr)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set_title(title)
    ax.label_outer()

    # Save the figure
    plt.savefig(os.path.join(results_dir, '{}.png'.format(title)))
    plt.close(fig)  # Close the figure to free memory

    # Convert to dB
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

    fig, ax = plt.subplots(figsize=(10, 7))

    # Display Mel-spectrogram
    img = librosa.display.specshow(mel_spectrogram_db, x_axis='time', y_axis='mel', sr=sr)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set_title(title)
    ax.label_outer()

    # Save the figure
    plt.savefig(os.path.join(results_dir_db, '{}.png'.format(title)))
    plt.close(fig)  # Close the figure to free memory




# Define file paths for the three different instruments and pitches
files = {
    'guitar_21': './datasets/nsynth-test/audio/guitar_acoustic_010-021-127.wav',
    'guitar_63': './datasets/nsynth-test/audio/guitar_acoustic_010-063-127.wav',
    'guitar_107': './datasets/nsynth-test/audio/guitar_acoustic_010-107-100.wav',
    'keyboard_28': './datasets/nsynth-test/audio/keyboard_electronic_003-028-127.wav',
    'keyboard_63': './datasets/nsynth-test/audio/keyboard_electronic_003-063-127.wav',
    'keyboard_96': './datasets/nsynth-test/audio/keyboard_electronic_003-096-127.wav',
    'reed_33': './datasets/nsynth-test/audio/reed_acoustic_023-033-127.wav',
    'reed_62': './datasets/nsynth-test/audio/reed_acoustic_023-062-100.wav',
    'reed_97': './datasets/nsynth-test/audio/reed_acoustic_023-097-127.wav'
}


def main():
    # Create the results directory if it doesn't exist
    results_dir = './results_mel'
    results_db_dir = './results_mel_db'
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(results_db_dir, exist_ok=True)

    # Iterate through the file paths and plot
    for i, (title, file_path) in enumerate(files.items()):
        y, sr = librosa.load(file_path)

        plot_mel_spectrogram(y, sr, title, results_dir, results_db_dir)
        print('{}: {} done!'.format(i, title))

    print("All Mel-spectrograms saved successfully.")


if __name__ == '__main__':
    main()
