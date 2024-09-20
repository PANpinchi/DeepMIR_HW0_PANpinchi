import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio

class NsynthDataset(Dataset):
    def __init__(self, audio_dir, json_file, transform=None):
        self.audio_dir = audio_dir
        self.transform = transform
        # Load metadata from examples.json
        with open(json_file, 'r') as f:
            self.metadata = json.load(f)
        self.keys = list(self.metadata.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        # Get audio filename and corresponding metadata
        note_str = self.keys[idx]
        file_name = f"{note_str}.wav"
        audio_path = os.path.join(self.audio_dir, file_name)

        # Load audio using torchaudio
        waveform, sample_rate = torchaudio.load(audio_path)

        # Get the label (qualities) from metadata
        label = torch.tensor(self.metadata[note_str]['instrument_family'], dtype=torch.long)

        if self.transform:
            waveform = self.transform(waveform)

        return waveform, label

def main():
    # Dataset and DataLoader initialization
    audio_dir = './datasets/nsynth-test/audio'
    json_file = './datasets/nsynth-test/examples.json'

    nsynth_dataset = NsynthDataset(audio_dir, json_file)
    nsynth_dataloader = DataLoader(nsynth_dataset, batch_size=32, shuffle=True, num_workers=4)

    # Iterate through DataLoader
    for batch_idx, (audio, labels) in enumerate(nsynth_dataloader):
        print(f"Batch {batch_idx} - Audio shape: {audio.shape}, Labels shape: {labels.shape}")

if __name__ == '__main__':
    main()
