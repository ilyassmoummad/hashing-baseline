import os
import json
import torch
import soundfile as sf
from torch.utils.data import Dataset


class VocalSoundDataset(Dataset):
    CLASS_NAMES = ["laughter", "sigh", "cough", "throatclearing", "sneeze", "sniff"]
    CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASS_NAMES)}
    SPLIT_TO_FILE = {
        "train": "tr.json",
        "val": "val.json",
        "test": "te.json"
    }

    def __init__(self, root_dir, split="train", transform=None, target_duration=5.0, sample_rate=16000):
        if split not in self.SPLIT_TO_FILE:
            raise ValueError(f"split must be one of {list(self.SPLIT_TO_FILE.keys())}")

        self.audio_dir = os.path.join(root_dir, "audio_16k")
        json_path = os.path.join(root_dir, "datafiles", self.SPLIT_TO_FILE[split])
        self.transform = transform
        self.sample_rate = sample_rate
        self.target_length = int(target_duration * sample_rate)

        with open(json_path, "r") as f:
            data = json.load(f)["data"]

        self.samples = []
        for item in data:
            filename = os.path.basename(item["wav"])
            label_str = filename.split("_")[-1].replace(".wav", "")
            if label_str not in self.CLASS_TO_IDX:
                raise ValueError(f"Unknown label '{label_str}' in {filename}")
            self.samples.append({
                "path": os.path.join(self.audio_dir, filename),
                "label": self.CLASS_TO_IDX[label_str]
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        waveform, sr = sf.read(sample["path"], dtype="float32")  # (samples, channels)
        
        # If stereo or multi-channel, convert to mono by averaging channels
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        
        waveform = torch.from_numpy(waveform)  # (samples,)

        # Pad/truncate to fixed length
        if waveform.size(0) > self.target_length:
            waveform = waveform[:self.target_length]
        elif waveform.size(0) < self.target_length:
            pad_amount = self.target_length - waveform.size(0)
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
        
        if self.transform:
            waveform = self.transform(waveform)

        return waveform, sample["label"]


class CREMADataset(Dataset):
    LABEL_MAP = {
        "N": 0,  # Neutral
        "H": 1,  # Happy
        "S": 2,  # Sad
        "A": 3,  # Angry
        "F": 4,  # Fear
        "D": 5,  # Disgust
    }
    TARGET_SR = 16000
    TARGET_LENGTH = 3 * TARGET_SR  # 3 seconds in samples

    def __init__(self, root_dir, split="train", transform=None, label_key="label"):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.label_key = label_key

        self.split_dir = os.path.join(root_dir, split)
        if not os.path.isdir(self.split_dir):
            raise ValueError(f"Split directory does not exist: {self.split_dir}")

        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        for fname in os.listdir(self.split_dir):
            if not fname.endswith(".wav"):
                continue

            wav_path = os.path.join(self.split_dir, fname)
            json_path = os.path.join(self.split_dir, fname.replace(".wav", ".json"))

            if not os.path.isfile(json_path):
                raise FileNotFoundError(f"JSON file not found for audio: {wav_path}")

            with open(json_path, "r") as f:
                metadata = json.load(f)

            label_str = metadata.get(self.label_key)
            if label_str is None:
                raise ValueError(f"Label key '{self.label_key}' not found in JSON: {json_path}")

            label = self.LABEL_MAP.get(label_str.upper())
            if label is None:
                raise ValueError(f"Unknown label '{label_str}' found in {json_path}")

            samples.append({
                "wav_path": wav_path,
                "label": label
            })

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        waveform, sr = sf.read(sample["wav_path"], dtype="float32")

        # Convert to mono if needed
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)

        # Resample if sample rate != TARGET_SR (optional)
        if sr != self.TARGET_SR:
            raise RuntimeError(f"Expected sample rate {self.TARGET_SR}, but got {sr}")

        # Pad or truncate to fixed length
        length = len(waveform)
        if length < self.TARGET_LENGTH:
            padding = self.TARGET_LENGTH - length
            waveform = torch.cat([torch.from_numpy(waveform), torch.zeros(padding)])
        else:
            waveform = torch.from_numpy(waveform[:self.TARGET_LENGTH])

        if self.transform:
            waveform = self.transform(waveform)

        return waveform, sample["label"]
    

class GTZANDataset(Dataset):
    LABEL_MAP = {
        "classical": 0,
        "jazz": 1,
        "pop": 2,
        "rock": 3,
        "hiphop": 4,
        "metal": 5,
        "blues": 6,
        "reggae": 7,
        "country": 8,
        "disco": 9,
    }
    TARGET_SR = 22050
    TARGET_LENGTH = 3 * TARGET_SR

    def __init__(self, root_dir, folds, transform=None, label_key="genre"):
        """
        Args:
            root_dir (str): Root directory containing GTZAN fold folders.
            folds (list[int|str]): List of fold numbers (e.g., [0,1,2]) or full fold names.
            transform (callable, optional): Optional transform to apply to each waveform.
            label_key (str): JSON key to extract the label from.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.label_key = label_key

        # Normalize folds to folder names
        self.folds = []
        for fold in folds:
            if isinstance(fold, int):
                self.folds.append(f"gtzan_fold_{fold}_0000000")
            elif isinstance(fold, str):
                # Allow passing already-correct folder names
                self.folds.append(fold)
            else:
                raise TypeError(f"Fold must be int or str, but got {type(fold).__name__}")

        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        for fold in self.folds:
            fold_path = os.path.join(self.root_dir, fold)
            if not os.path.isdir(fold_path):
                raise ValueError(f"Fold directory does not exist: {fold_path}")

            for fname in os.listdir(fold_path):
                if not fname.endswith(".wav"):
                    continue

                wav_path = os.path.join(fold_path, fname)
                json_path = wav_path.replace(".wav", ".json")

                if not os.path.isfile(json_path):
                    raise FileNotFoundError(f"JSON file not found for audio: {wav_path}")

                with open(json_path, "r") as f:
                    metadata = json.load(f)

                label_str = metadata.get(self.label_key)
                if label_str is None:
                    raise ValueError(f"Label key '{self.label_key}' not found in JSON: {json_path}")

                label = self.LABEL_MAP.get(label_str.lower())
                if label is None:
                    raise ValueError(f"Unknown label '{label_str}' found in {json_path}")

                samples.append({
                    "wav_path": wav_path,
                    "label": label
                })

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        waveform, sr = sf.read(sample["wav_path"], dtype="float32")

        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)

        if sr != self.TARGET_SR:
            raise RuntimeError(f"Expected sample rate {self.TARGET_SR}, but got {sr}")

        length = len(waveform)
        if length < self.TARGET_LENGTH:
            padding = self.TARGET_LENGTH - length
            waveform = torch.cat([torch.from_numpy(waveform), torch.zeros(padding)])
        else:
            waveform = torch.from_numpy(waveform[:self.TARGET_LENGTH])

        if self.transform:
            waveform = self.transform(waveform)

        return waveform, sample["label"]