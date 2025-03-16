import os
import cv2
import torch
import torchaudio
import subprocess
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import MelSpectrogram
from transformers import AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class MELDDataset(Dataset):
    def __init__(self, csv_path, video_dir):
        self.csv_path = csv_path
        self.video_dir = video_dir
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        # Create a mapping of emotions to integers
        #  Anger, Disgust, Sadness, Joy, Neutral, Surprise and Fear
        self.emotion_map = {
            "anger": 0,
            "disgust": 1,
            "fear": 2,
            "joy": 3,
            "neutral": 4,
            "sadness": 5,
            "surprise": 6
        }

        # Create a mapping of sentiment to integers
        #  Positive, Negative, Neutral
        self.sentiment_map = {
            "negative": 0,
            "neutral": 1,
            "positive": 2
        }

        # Load the data
        abs_path = os.path.abspath(csv_path)
        print(f"Attempting to read file from: {abs_path}")

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"File not found: {abs_path}")

        self.data = pd.read_csv(csv_path)
        print("Data loaded Successfully from: ", abs_path)
        print(self.data.head())

        print("Csv Length: ", len(self.data))

    def _load_video_frames(self, video_path):
        frames = []

        # Load the video using OpenCV
        cap = cv2.VideoCapture(video_path)

        try:
            # Check if the video was loaded successfully
            if not cap.isOpened():
                raise ValueError(f"Error loading video: {video_path}")

            # Try to read the first frame from the video -> Validate the video
            # ret = True if the frame was read successfully
            ret, frame = cap.read()

            if not ret or frame is None:
                raise ValueError(f"Error loading video: {video_path}")

            while ret:
                frames.append(frame)
                ret, frame = cap.read()

            # -- Important --
            # Reset the frames to the beginning of the video
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            # -- Important --
            # You increase the no of frames more than 30 in case needed - for NN Model Training purposes -
            while len(frames) < 30 and cap.isOpened():
                ret, frame = cap.read()

                if not ret or frame is None:
                    break

                # -- Important --
                # Resize the frame to 224x224
                frame = cv2.resize(frame, (224, 224))

                # Normalize the frame RGB Channels to [0, 1]
                # LLM Standardization -> (X - mean) / std -> (X - 128) / 128 -> X / 255 (In Computer Vision)
                # [255, 255, 255] -> [1, 1, 1]
                # [128, 150, 100] -> [1, 0.5, 0]
                frame = frame / 255.0
                frames.append(frame)

        except Exception as e:
            raise ValueError(f"Error loading video: {str(e)}")
        finally:
            cap.release()

        if (len(frames) == 0):
            raise ValueError(f"Video Frames Extraction Failed...!")

        # Add Padding to the frames if the video is less than 30 frames
        if len(frames) < 30:
            frames += [np.zeros_like(frames[0]) * (30 - len(frames))]
        else:
            # Trim the frames if the video is more than 30 frames
            frames = frames[:30]

        # Convert the frames to a float tensor + permute the dimensions
        # Before the permute -> [frames, height, width, channels] -> [30, 224, 224, 3]
        ###  ======= >>> [0, 1, 2, 3] -> [0, 3, 1, 2] <<< ======= ###
        # After the permute -> [frames, channels, height, width] -> [30, 3, 224, 224]
        return torch.FloatTensor(np.array(frames)).permute(0, 3, 1, 2)

    def _extract_audio_features(self, video_path):

        # Replace the video extension with the audio extension
        audio_path = video_path.replace(".mp4", ".wav")

        try:
            # Extract the audio features using FFmpeg
            print("Extracting Audio Features...")
            subprocess.run(["ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            print("Audio Features Extracted Successfully...!")

            # Load the audio file using torchaudio
            waveform, sample_rate = torchaudio.load(audio_path)

            # Check if the sample rate is 16KHz
            # Speech Recognition Models require a sample rate of 16KHz -- 16000 samples per second --
            if sample_rate != 16000:
                # raise ValueError(f"Invalid Sample Rate: {sample_rate}")
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)

            # MEL Spectrogram Parameters -> 64 MEL Bands, 1024 FFT Size, 512 Hop Length (Stride)
            # MEL Spectrogram -> 2D Tensor -> [Time on X-Axis, Frequency on Y-Axis] -- Standard Audio Feature Representation --
            mel_spectrogram = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000,
                n_mels=64,
                n_fft=1024,
                hop_length=512
            )

            # Extract the MEL Spectrogram
            mel_spectrogram = mel_spectrogram(waveform)

            # Normalize the MEL Spectrogram using Standardization
            # -> (X - mean) / std
            # -> (X - X.mean()) / X.std()
            # -> Z-Score Normalization -- Standardization --
            mel_spectrogram = (mel_spectrogram -
                               mel_spectrogram.mean()) / mel_spectrogram.std()

            # Add Padding to the MEL Spectrogram if the no of frames is less than 300
            if mel_spectrogram.size(2) < 300:
                padding = 300 - mel_spectrogram.size(2)
                mel_spectrogram = torch.nn.functional.pad(
                    mel_spectrogram, (0, padding))
            else:
                # Trim the MEL Spectrogram if the no of frames is more than 300
                mel_spectrogram = mel_spectrogram[:, :, :300]

            # Return the MEL Spectrogram
            return mel_spectrogram

        except subprocess.CalledProcessError as e:
            raise ValueError(f"Error extracting audio features: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error Loading Audio File: {str(e)}")
        finally:
            # Remove the audio file after extracting the features -- Clean Up --
            if os.path.exists(audio_path):
                os.remove(audio_path)

    # Override the __len__ method

    def __len__(self):
        return len(self.data)

    # Override the __getitem__ method

    def __getitem__(self, index):
        #
        if isinstance(index, torch.Tensor):
            index = index.item()

        # Get the row at the given index
        row = self.data.iloc[index]

        try:
            # Get the video filename + path
            video_filename = f"""dia{row["Dialogue_ID"]}_utt{row["Utterance_ID"]}.mp4"""

            # Check if the video file exists
            path = os.path.join(self.video_dir, video_filename)
            video_path = os.path.exists(path)

            if video_path == False:
                raise FileNotFoundError(f"File not found: {path}")

            print(f"Video Path: {path}")

            # Tokenize the text input using the BERT tokenizer
            # Max Length = 128 -- This is the maximum length of the input sequence (No of tokens) --
            text_inputs = self.tokenizer(
                row["Utterance"], padding="max_length", truncation=True, max_length=128, return_tensors="pt")

            # Extract the video frames
            video_frames = self._load_video_frames(path)

            # Extract the audio features
            audio_features = self._extract_audio_features(path)

            # Map Sentiment + Emotion Labels to Integers -- Classification Task --
            emotion_label = self.emotion_map[row["Emotion"].lower()]
            sentiment_label = self.sentiment_map[row["Sentiment"].lower()]

            # Print the data -- Debugging --
            # print(text_inputs)
            # print(video_frames)
            # print(audio_features)

            return {
                "text_inputs": {
                    # Text Input IDs -> Tensor of Token IDs -- Input to the BERT Model --
                    # Shape: [1, 128] -- Batch Size = 1, Sequence Length = 128 --
                    "input_ids": text_inputs["input_ids"].squeeze(),
                    "attention_mask": text_inputs["attention_mask"].squeeze(0)
                },
                "video_frames": video_frames,
                "audio_features": audio_features,
                # Emotion + Sentiment Labels -> Tensor of Integers -- Target Labels --
                "emotion_label": torch.tensor(emotion_label),
                "sentiment_label": torch.tensor(sentiment_label)
            }

        # Handle any exceptions that occur during the data processing
        except Exception as e:
            print(f"Error Processing Row {path}: {str(e)}")
            return None


def collate_fn(batch):
    # Remove any None values from the batch -- Error Handling --
    batch = list(filter(None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def prepare_dataloaders(train_csv, train_video_dir, dev_csv, dev_video_dir, test_csv, test_video_dir, batch_size=32):
    # Create the Training, Validation and Testing Datasets
    train_dataset = MELDDataset(train_csv, train_video_dir)
    dev_dataset = MELDDataset(dev_csv, dev_video_dir)
    test_dataset = MELDDataset(test_csv, test_video_dir)

    # Create the Training, Validation and Testing Data Loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(
        dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, dev_loader, test_loader


if __name__ == "__main__":
    # Create the Data Loaders for the MELD Dataset -- Taining, Validation and Testing --
    train_loder, dev_loader, test_loader = prepare_dataloaders("../Dataset/MELD.Raw/train/train_sent_emo.csv",
                                                               "../Dataset/MELD.Raw/train/train_splits",
                                                               "../Dataset/MELD.Raw/dev/dev_sent_emo.csv",
                                                               "../Dataset/MELD.Raw/dev/dev_splits_complete",
                                                               "../Dataset/MELD.Raw/test/test_sent_emo.csv",
                                                               "../Dataset/MELD.Raw/test/output_repeated_splits_test")
"""    
    # Iterate over the training loader
    for batch in train_loder:
        # print(batch)
        print(batch["text_inputs"])
        print(batch["video_frames"].shape)
        print(batch["audio_features"].shape)
        print(batch["emotion_label"])
        print(batch["sentiment_label"])
        break

    # Iterate over the validation loader
    for batch in dev_loader:
        # print(batch)
        print(batch["text_inputs"])
        print(batch["video_frames"].shape)
        print(batch["audio_features"].shape)
        print(batch["emotion_label"])
        print(batch["sentiment_label"])
        break

    # Iterate over the testing loader
    for batch in test_loader:
        # print(batch)
        print(batch["text_inputs"])
        print(batch["video_frames"].shape)
        print(batch["audio_features"].shape)
        print(batch["emotion_label"])
        print(batch["sentiment_label"])
        break
"""