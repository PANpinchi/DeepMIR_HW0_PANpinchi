import os
import json
import numpy as np
import librosa
from tqdm import tqdm


def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    return np.mean(log_mel_spec.T, axis=0)


def load_dataset(json_path, audio_dir):
    with open(json_path, 'r') as f:
        data = json.load(f)

    features = []
    labels = []

    for key, value in tqdm(data.items()):
        audio_path = os.path.join(audio_dir, f"{key}.wav")
        if os.path.exists(audio_path):
            feature = extract_features(audio_path)
            features.append(feature)
            labels.append(value["instrument_family"])

    return np.array(features), np.array(labels)


def save_preprocessed_data(json_path, audio_dir, output_dir, name):
    X, y = load_dataset(json_path, audio_dir)
    np.save(os.path.join(output_dir, 'X_{}.npy'.format(name)), X)
    np.save(os.path.join(output_dir, 'y_{}.npy'.format(name)), y)
    print(f"Preprocessed {name} data saved to {output_dir}")


if __name__ == '__main__':
    subtrain_audio_dir = './datasets/nsynth-subtrain/audio'
    subtrain_json_file = './datasets/nsynth-subtrain/examples.json'
    test_audio_dir = './datasets/nsynth-test/audio'
    test_json_file = './datasets/nsynth-test/examples.json'

    output_dir = './datasets/'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    save_preprocessed_data(test_json_file, test_audio_dir, output_dir, name='test')
    save_preprocessed_data(subtrain_json_file, subtrain_audio_dir, output_dir, name='subtrain')
    
