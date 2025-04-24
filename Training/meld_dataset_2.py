import os
import cv2
import torch
import torchaudio
import subprocess
import numpy as np
import pandas as pd
import hashlib
import logging
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import MelSpectrogram
from transformers import AutoTokenizer
from torchvision import transforms

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MELDDataset")

# Prevent tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class MELDDataset(Dataset):
    def __init__(self, 
                 csv_path, 
                 video_dir, 
                 cache_dir=None,
                 video_frames=30, 
                 audio_frames=300,
                 frame_size=(224, 224), 
                 mel_bands=64,
                 is_training=False,
                 preprocess_audio=False):
        """
        Initialize the MELD Dataset
        
        Args:
            csv_path (str): Path to the CSV file containing the dataset
            video_dir (str): Path to the directory containing the videos
            cache_dir (str): Path to cache extracted features (None for no caching)
            video_frames (int): Number of video frames to extract
            audio_frames (int): Number of audio frames to extract
            frame_size (tuple): Size to resize video frames to (height, width)
            mel_bands (int): Number of MEL bands for audio processing
            is_training (bool): Whether this dataset is for training (enables augmentations)
            preprocess_audio (bool): Whether to pre-extract all audio files
        """
        self.csv_path = csv_path
        self.video_dir = video_dir
        self.cache_dir = cache_dir
        self.video_frames = video_frames
        self.audio_frames = audio_frames
        self.frame_size = frame_size
        self.mel_bands = mel_bands
        self.is_training = is_training
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        # Create cache directories if needed
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            os.makedirs(os.path.join(cache_dir, "audio"), exist_ok=True)
            os.makedirs(os.path.join(cache_dir, "video"), exist_ok=True)
            os.makedirs(os.path.join(cache_dir, "mel"), exist_ok=True)

        # Create a mapping of emotions to integers
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
        self.sentiment_map = {
            "negative": 0,
            "neutral": 1,
            "positive": 2
        }

        # Error tracking
        self.error_counts = {
            "video_not_found": 0,
            "video_load_error": 0,
            "audio_extract_error": 0,
            "other_error": 0
        }
        
        # Load the data
        abs_path = os.path.abspath(csv_path)
        logger.info(f"Attempting to read file from: {abs_path}")

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"File not found: {abs_path}")

        self.data = pd.read_csv(csv_path)
        logger.info(f"Data loaded Successfully from: {abs_path}")
        logger.info(f"Dataset contains {len(self.data)} samples")
        
        # Data augmentation for training
        if is_training:
            self.video_transforms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
            ])
        else:
            self.video_transforms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
            ])
            
        # Pre-extract audio if requested
        if preprocess_audio:
            self._preprocess_audio_files()

    def _preprocess_audio_files(self):
        """Pre-extract all audio files to avoid on-the-fly extraction during training"""
        logger.info(f"Pre-extracting audio files for {len(self.data)} videos...")
        audio_dir = os.path.join(self.cache_dir, "audio") if self.cache_dir else "audio_cache"
        
        if not os.path.exists(audio_dir):
            os.makedirs(audio_dir)
            
        # Create a list of videos to process
        videos_to_process = []
        for idx, row in self.data.iterrows():
            video_filename = f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.mp4"
            video_path = os.path.join(self.video_dir, video_filename)
            audio_path = os.path.join(audio_dir, video_filename.replace(".mp4", ".wav"))
            
            if not os.path.exists(audio_path) and os.path.exists(video_path):
                videos_to_process.append((video_path, audio_path))
                
        # Process videos in parallel
        with ThreadPoolExecutor(max_workers=min(8, multiprocessing.cpu_count())) as executor:
            list(executor.map(self._extract_audio_file, videos_to_process))
            
        logger.info(f"Audio extraction completed. {len(videos_to_process)} files processed.")
        
    def _extract_audio_file(self, paths):
        """Extract audio file from video"""
        video_path, audio_path = paths
        try:
            subprocess.run(
                ["ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le", 
                 "-ar", "16000", "-ac", "1", audio_path],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            return True
        except subprocess.CalledProcessError as e:
            logger.warning(f"Error extracting audio from {video_path}: {e}")
            self.error_counts["audio_extract_error"] += 1
            return False
        
    def _get_cache_path(self, file_path, feature_type):
        """Generate a cache path for features"""
        if not self.cache_dir:
            return None
            
        cache_key = hashlib.md5(file_path.encode()).hexdigest()
        return os.path.join(self.cache_dir, feature_type, f"{cache_key}.pt")
        
    def _load_video_frames(self, video_path):
        """Load and process video frames"""
        cache_path = self._get_cache_path(video_path, "video")
        if cache_path and os.path.exists(cache_path):
            return torch.load(cache_path)
        
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        try:
            if not cap.isOpened():
                self.error_counts["video_load_error"] += 1
                raise ValueError(f"Error loading video: {video_path}")
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            logger.info(f"Processing video {video_path}: {total_frames} frames")
            
            if total_frames > self.video_frames * 2:
                sample_indices = np.linspace(0, total_frames-1, self.video_frames, dtype=int)
                for idx in sample_indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        if len(frame.shape) == 2 or frame.shape[2] == 1:
                            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                        else:
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame = cv2.resize(frame, self.frame_size)
                        frame = frame / 255.0
                        frames.append(frame)
            else:
                while len(frames) < self.video_frames and cap.isOpened():
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        break
                    if len(frame.shape) == 2 or frame.shape[2] == 1:
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                    else:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, self.frame_size)
                    frame = frame / 255.0
                    frames.append(frame)
            
            if len(frames) == 0:
                self.error_counts["video_load_error"] += 1
                raise ValueError(f"Video Frames Extraction Failed...!")
            
            if len(frames) < self.video_frames:
                frames += [np.zeros_like(frames[0])] * (self.video_frames - len(frames))
            else:
                frames = frames[:self.video_frames]
            
            frames_tensor = torch.FloatTensor(np.array(frames)).permute(0, 3, 1, 2)
            logger.info(f"Frames tensor shape before augmentation: {frames_tensor.shape}")
            assert frames_tensor.shape[1] == 3, f"Expected 3 channels, got {frames_tensor.shape[1]}"
            
            if self.is_training and self.video_transforms:
                augmented_frames = []
                for i in range(frames_tensor.size(0)):
                    frame = frames_tensor[i]  # CHW format
                    augmented = self.video_transforms(frame)
                    augmented_frames.append(augmented)
                frames_tensor = torch.stack(augmented_frames)
            
            if cache_path:
                torch.save(frames_tensor, cache_path)
            
            return frames_tensor
        
        except Exception as e:
            self.error_counts["video_load_error"] += 1
            logger.warning(f"Error loading video {video_path}: {str(e)}")
            raise ValueError(f"Error loading video: {str(e)}")
        finally:
            cap.release()

    def _extract_audio_features(self, video_path):
        """Extract audio features from video or pre-extracted audio file"""
        cache_path = self._get_cache_path(video_path, "mel")
        if cache_path and os.path.exists(cache_path):
            return torch.load(cache_path)
            
        audio_dir = os.path.join(self.cache_dir, "audio") if self.cache_dir else None
        if audio_dir and os.path.exists(audio_dir):
            audio_path = os.path.join(audio_dir, os.path.basename(video_path).replace(".mp4", ".wav"))
        else:
            audio_path = video_path.replace(".mp4", ".wav")
            
        try:
            if not os.path.exists(audio_path):
                logger.info(f"Extracting audio from {video_path}")
                subprocess.run(
                    ["ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le", 
                     "-ar", "16000", "-ac", "1", audio_path],
                    check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                logger.info("Audio Features Extracted Successfully")

            waveform, sample_rate = torchaudio.load(audio_path)

            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)

            mel_spectrogram = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000,
                n_mels=self.mel_bands,
                n_fft=1024,
                hop_length=512
            )

            mel_specs = mel_spectrogram(waveform)

            if mel_specs.numel() == 0 or mel_specs.std() == 0:
                logger.warning(f"Empty or silent audio in {video_path}")
                mel_specs = torch.zeros((1, self.mel_bands, self.audio_frames))
            else:
                mel_specs = (mel_specs - mel_specs.mean()) / (mel_specs.std() + 1e-8)

            if mel_specs.size(2) < self.audio_frames:
                padding = self.audio_frames - mel_specs.size(2)
                mel_specs = torch.nn.functional.pad(mel_specs, (0, padding))
            else:
                mel_specs = mel_specs[:, :, :self.audio_frames]
                
            if cache_path:
                torch.save(mel_specs, cache_path)

            return mel_specs

        except subprocess.CalledProcessError as e:
            self.error_counts["audio_extract_error"] += 1
            logger.error(f"Error extracting audio features: {str(e)}")
            raise ValueError(f"Error extracting audio features: {str(e)}")
        except Exception as e:
            self.error_counts["audio_extract_error"] += 1
            logger.error(f"Error loading audio file: {str(e)}")
            raise ValueError(f"Error loading audio file: {str(e)}")
        finally:
            if os.path.exists(audio_path) and audio_dir not in audio_path:
                os.remove(audio_path)

    def report_error_stats(self):
        """Report error statistics"""
        total = sum(self.error_counts.values())
        if total > 0:
            logger.info(f"Encountered {total} errors out of {len(self.data)} samples:")
            for error_type, count in self.error_counts.items():
                logger.info(f"  - {error_type}: {count} ({count/len(self.data)*100:.2f}%)")
        else:
            logger.info("No errors encountered during dataset processing")
            
    def analyze_video_statistics(self, sample_size=100):
        """Analyze video statistics to help determine optimal parameters"""
        logger.info(f"Analyzing video statistics (sample size: {sample_size})...")
        
        if len(self.data) > sample_size:
            sample_indices = np.random.choice(len(self.data), sample_size, replace=False)
            sample_data = self.data.iloc[sample_indices]
        else:
            sample_data = self.data
            
        frame_counts = []
        durations = []
        resolutions = []
        
        for _, row in sample_data.iterrows():
            try:
                video_filename = f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.mp4"
                video_path = os.path.join(self.video_dir, video_filename)
                
                if os.path.exists(video_path):
                    cap = cv2.VideoCapture(video_path)
                    if cap.isOpened():
                        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        
                        frame_counts.append(frame_count)
                        durations.append(frame_count / fps if fps > 0 else 0)
                        resolutions.append((width, height))
                        
                    cap.release()
            except Exception as e:
                logger.warning(f"Error analyzing video {video_path}: {e}")
                
        if frame_counts:
            logger.info(f"Video frame count statistics:")
            logger.info(f"  Min frames: {min(frame_counts)}")
            logger.info(f"  Max frames: {max(frame_counts)}")
            logger.info(f"  Mean frames: {np.mean(frame_counts):.2f}")
            logger.info(f"  Median frames: {np.median(frame_counts)}")
            logger.info(f"  95th percentile: {np.percentile(frame_counts, 95)}")
            
        if durations:
            logger.info(f"Video duration statistics (seconds):")
            logger.info(f"  Min duration: {min(durations):.2f}")
            logger.info(f"  Max duration: {max(durations):.2f}")
            logger.info(f"  Mean duration: {np.mean(durations):.2f}")
            logger.info(f"  Median duration: {np.median(durations):.2f}")
            
        if resolutions:
            res_count = {}
            for res in resolutions:
                if res in res_count:
                    res_count[res] += 1
                else:
                    res_count[res] = 1
                    
            logger.info(f"Video resolution statistics:")
            for res, count in sorted(res_count.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"  {res[0]}x{res[1]}: {count} videos ({count/len(resolutions)*100:.2f}%)")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """Get a sample from the dataset with error handling"""
        if isinstance(index, torch.Tensor):
            index = index.item()

        row = self.data.iloc[index]
        
        try:
            video_filename = f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.mp4"
            path = os.path.join(self.video_dir, video_filename)
            
            if not os.path.exists(path):
                self.error_counts["video_not_found"] += 1
                raise FileNotFoundError(f"File not found: {path}")

            text_inputs = self.tokenizer(
                row["Utterance"], 
                padding="max_length", 
                truncation=True, 
                max_length=128, 
                return_tensors="pt"
            )

            video_frames = self._load_video_frames(path)
            audio_features = self._extract_audio_features(path)

            emotion_label = self.emotion_map[row["Emotion"].lower()]
            sentiment_label = self.sentiment_map[row["Sentiment"].lower()]

            return {
                "text_inputs": {
                    "input_ids": text_inputs["input_ids"].squeeze(),
                    "attention_mask": text_inputs["attention_mask"].squeeze(0)
                },
                "video_frames": video_frames,
                "audio_features": audio_features,
                "emotion_label": torch.tensor(emotion_label),
                "sentiment_label": torch.tensor(sentiment_label),
                "metadata": {
                    "dialogue_id": row["Dialogue_ID"],
                    "utterance_id": row["Utterance_ID"],
                    "utterance": row["Utterance"]
                }
            }

        except FileNotFoundError as e:
            logger.warning(f"File not found error: {str(e)}")
            self.error_counts["video_not_found"] += 1
            return self._create_default_sample()
                
        except cv2.error as e:
            logger.warning(f"OpenCV error processing video: {str(e)}")
            self.error_counts["video_load_error"] += 1
            return self._create_default_sample()
                
        except Exception as e:
            logger.error(f"Error processing sample {index}: {str(e)}")
            self.error_counts["other_error"] += 1
            return self._create_default_sample()
                
    def _create_default_sample(self):
        """Create a default sample for error cases"""
        text_inputs = self.tokenizer(
            "error placeholder", 
            padding="max_length", 
            truncation=True, 
            max_length=128, 
            return_tensors="pt"
        )
        
        return {
            "text_inputs": {
                "input_ids": text_inputs["input_ids"].squeeze(),
                "attention_mask": text_inputs["attention_mask"].squeeze(0)
            },
            "video_frames": torch.zeros((self.video_frames, 3, *self.frame_size)),
            "audio_features": torch.zeros((1, self.mel_bands, self.audio_frames)),
            "emotion_label": torch.tensor(self.emotion_map["neutral"]),
            "sentiment_label": torch.tensor(self.sentiment_map["neutral"]),
            "metadata": {
                "dialogue_id": -1,
                "utterance_id": -1,
                "utterance": "error placeholder"
            }
        }


def collate_fn(batch):
    """Custom collate function with dynamic padding and error handling"""
    batch = list(filter(None, batch))
    
    if not batch:
        raise ValueError("Batch is empty after filtering")
        
    return torch.utils.data.dataloader.default_collate(batch)


def prepare_dataloaders(train_csv, train_video_dir, dev_csv, dev_video_dir, 
                        test_csv, test_video_dir, batch_size=32, cache_dir=None,
                        num_workers=4, video_frames=30, audio_frames=300,
                        preprocess_audio=True):
    """
    Create train, validation and test dataloaders with optimized settings
    
    Args:
        train_csv (str): Path to training CSV file
        train_video_dir (str): Path to training video directory
        dev_csv (str): Path to validation CSV file
        dev_video_dir (str): Path to validation video directory
        test_csv (str): Path to test CSV file
        test_video_dir (str): Path to test video directory
        batch_size (int): Batch size for dataloaders
        cache_dir (str): Path to cache directory
        num_workers (int): Number of worker processes for data loading
        video_frames (int): Number of video frames to extract
        audio_frames (int): Number of audio frames to extract
        preprocess_audio (bool): Whether to pre-extract all audio files
        
    Returns:
        tuple: (train_loader, dev_loader, test_loader)
    """
    
    train_cache = os.path.join(cache_dir, "train") if cache_dir else None
    dev_cache = os.path.join(cache_dir, "dev") if cache_dir else None  
    test_cache = os.path.join(cache_dir, "test") if cache_dir else None
    
    logger.info("Creating training dataset...")
    train_dataset = MELDDataset(
        train_csv, train_video_dir, 
        cache_dir=train_cache,
        video_frames=video_frames, 
        audio_frames=audio_frames,
        is_training=True,
        preprocess_audio=preprocess_audio
    )
    
    train_dataset.analyze_video_statistics()
    
    logger.info("Creating validation dataset...")
    dev_dataset = MELDDataset(
        dev_csv, dev_video_dir, 
        cache_dir=dev_cache,
        video_frames=video_frames, 
        audio_frames=audio_frames,
        is_training=False,
        preprocess_audio=preprocess_audio
    )
    
    logger.info("Creating test dataset...")
    test_dataset = MELDDataset(
        test_csv, test_video_dir, 
        cache_dir=test_cache,
        video_frames=video_frames, 
        audio_frames=audio_frames,
        is_training=False,
        preprocess_audio=preprocess_audio
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    dev_loader = DataLoader(
        dev_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    logger.info("Error statistics for training set:")
    train_dataset.report_error_stats()
    
    logger.info("Error statistics for validation set:")
    dev_dataset.report_error_stats()
    
    logger.info("Error statistics for test set:")
    test_dataset.report_error_stats()

    return train_loader, dev_loader, test_loader


if __name__ == "__main__":
    train_loader, dev_loader, test_loader = prepare_dataloaders(
        "../Dataset/MELD.Raw/train/train_sent_emo.csv",
        "../Dataset/MELD.Raw/train/train_splits",
        "../Dataset/MELD.Raw/dev/dev_sent_emo.csv",
        "../Dataset/MELD.Raw/dev/dev_splits_complete",
        "../Dataset/MELD.Raw/test/test_sent_emo.csv",
        "../Dataset/MELD.Raw/test/output_repeated_splits_test",
        batch_size=16,
        cache_dir="./meld_cache",
        num_workers=4,
        video_frames=30,
        audio_frames=300,
        preprocess_audio=True
    )
    
    logger.info("Testing training dataloader...")
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx == 0:
            logger.info(f"Training batch shapes:")
            logger.info(f"  Text input IDs: {batch['text_inputs']['input_ids'].shape}")
            logger.info(f"  Video frames: {batch['video_frames'].shape}")
            logger.info(f"  Audio features: {batch['audio_features'].shape}")
            logger.info(f"  Emotion labels: {batch['emotion_label'].shape}")
            logger.info(f"  Sentiment labels: {batch['sentiment_label'].shape}")
        if batch_idx >= 2:
            break
            
    logger.info("\nTesting validation dataloader...")
    for batch_idx, batch in enumerate(dev_loader):
        if batch_idx == 0:
            logger.info(f"Validation batch shapes:")
            logger.info(f"  Text input IDs: {batch['text_inputs']['input_ids'].shape}")
            logger.info(f"  Video frames: {batch['video_frames'].shape}")
            logger.info(f"  Audio features: {batch['audio_features'].shape}")
        if batch_idx >= 2:
            break
            
    logger.info("\nTesting test dataloader...")
    for batch_idx, batch in enumerate(test_loader):
        if batch_idx == 0:
            logger.info(f"Test batch shapes:")
            logger.info(f"  Text input IDs: {batch['text_inputs']['input_ids'].shape}")
            logger.info(f"  Video frames: {batch['video_frames'].shape}")
            logger.info(f"  Audio features: {batch['audio_features'].shape}")
        if batch_idx >= 2:
            break
            
    logger.info("All dataloaders tested successfully!")