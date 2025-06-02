import torch
from models import MultimodalSentimentModel
import os
import cv2
import numpy as np
import subprocess
import torchaudio
import whisper
from transformers import AutoTokenizer
import sys
import json
import boto3
import tempfile
import logging
import soundfile

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

EMOTION_MAP = {0: "anger", 1: "disgust", 2: "fear",
               3: "joy", 4: "neutral", 5: "sadness", 6: "surprise"}
SENTIMENT_MAP = {0: "negative", 1: "neutral", 2: "positive"}

def install_ffmpeg():
    logger.debug("Checking FFmpeg installation...")
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=True)
        logger.debug("FFmpeg version: %s", result.stdout)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.error("FFmpeg not found: %s", e)
        return False

class VideoProcessor:
    def process_video(self, video_path):
        logger.debug("Processing video: %s", video_path)
        cap = cv2.VideoCapture(video_path)
        frames = []

        try:
            if not cap.isOpened():
                raise ValueError(f"Video not found or cannot be opened: {video_path}")

            ret, frame = cap.read()
            if not ret or frame is None:
                raise ValueError(f"First frame read failed: {video_path}")

            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            while len(frames) < 30 and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.resize(frame, (224, 224))
                frame = frame / 255.0
                frames.append(frame)

        except Exception as e:
            raise ValueError(f"Video processing error: {str(e)}")
        finally:
            cap.release()

        if len(frames) == 0:
            raise ValueError("No frames could be extracted")

        if len(frames) < 30:
            frames += [np.zeros_like(frames[0])] * (30 - len(frames))
        else:
            frames = frames[:30]

        logger.debug("Processed %d video frames", len(frames))
        return torch.FloatTensor(np.array(frames)).permute(0, 3, 1, 2)

class AudioProcessor:
    def extract_features(self, video_path, max_length=300):
        temp_dir = tempfile.gettempdir()
        audio_path = os.path.normpath(os.path.join(temp_dir, f"temp_audio_{os.getpid()}.wav"))

        logger.debug("Extracting audio from %s to %s", video_path, audio_path)
        try:
            # Use aac codec to ensure compatibility
            result = subprocess.run([
                'ffmpeg',
                '-i', os.path.normpath(video_path),
                '-vn', '-acodec', 'pcm_s16le',
                '-ar', '16000', '-ac', '1',
                '-y', audio_path
            ], check=True, capture_output=True, text=True)
            logger.debug("FFmpeg audio extraction output: %s", result.stderr)

            if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
                raise ValueError(f"Invalid or empty audio file: {audio_path}")

            # Validate WAV file
            try:
                with soundfile.SoundFile(audio_path) as sf:
                    logger.debug("WAV file: %s Hz, %s channels", sf.samplerate, sf.channels)
                    if sf.samplerate != 16000 or sf.channels != 1:
                        logger.warning("WAV file has unexpected format: %s Hz, %s channels", sf.samplerate, sf.channels)
            except Exception as e:
                logger.error("WAV validation failed: %s", e)
                raise ValueError(f"Invalid WAV file: {str(e)}")

            waveform, sample_rate = torchaudio.load(audio_path)

            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)

            mel_spectrogram = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000,
                n_mels=64,
                n_fft=1024,
                hop_length=512
            )

            mel_spec = mel_spectrogram(waveform)
            mel_spec = (mel_spec - mel_spec.mean()) / mel_spec.std()

            if mel_spec.size(2) < 300:
                padding = 300 - mel_spec.size(2)
                mel_spec = torch.nn.functional.pad(mel_spec, (0, padding))
            else:
                mel_spec = mel_spec[:, :, :300]

            logger.debug("Audio features shape: %s", mel_spec.shape)
            return mel_spec

        except subprocess.CalledProcessError as e:
            logger.error("Audio extraction error: %s\nFFmpeg stderr: %s", e, e.stderr)
            raise ValueError(f"Audio extraction error: {str(e)}")
        except Exception as e:
            logger.error("Audio processing error: %s", e)
            raise ValueError(f"Audio error: {str(e)}")
        finally:
            try:
                if os.path.exists(audio_path):
                    os.remove(audio_path)
            except Exception as e:
                logger.warning("Failed to clean up temporary audio file: %s", e)

class VideoUtteranceProcessor:
    def __init__(self):
        self.video_processor = VideoProcessor()
        self.audio_processor = AudioProcessor()

    def extract_segment(self, video_path, start_time, end_time, temp_dir=None):
        if temp_dir is None:
            temp_dir = tempfile.gettempdir()
        os.makedirs(temp_dir, exist_ok=True)

        segment_path = os.path.normpath(os.path.join(
            temp_dir, f"segment_{start_time}_{end_time}_{os.getpid()}.mp4"))

        logger.debug("Extracting segment from %s (%ss to %ss) to %s", video_path, start_time, end_time, segment_path)
        try:
            result = subprocess.run([
                "ffmpeg", "-i", os.path.normpath(video_path),
                "-ss", str(start_time), "-to", str(end_time),
                "-c:v", "libx264", "-c:a", "aac",
                "-y", segment_path
            ], check=True, capture_output=True, text=True)
            logger.debug("FFmpeg segment extraction output: %s", result.stderr)

            if not os.path.exists(segment_path) or os.path.getsize(segment_path) == 0:
                raise ValueError(f"Segment extraction failed: {segment_path}")

            return segment_path
        except subprocess.CalledProcessError as e:
            logger.error("Segment extraction error: %s\nFFmpeg stderr: %s", e, e.stderr)
            raise ValueError(f"Segment extraction failed: {str(e)}")
        except Exception as e:
            logger.error("Segment extraction error: %s", e)
            try:
                if os.path.exists(segment_path):
                    os.remove(segment_path)
            except Exception as cleanup_error:
                logger.warning("Failed to clean up temporary segment file: %s", cleanup_error)
            raise ValueError(f"Segment extraction failed: {str(e)}")

def get_video_duration(video_path):
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ]
        duration = float(subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode().strip())
        logger.debug("Video duration: %ss", duration)
        return duration
    except Exception as e:
        logger.error("Failed to get video duration: %s", e)
        return 30.0

def download_from_s3(s3_uri):
    s3_client = boto3.client("s3")
    bucket = s3_uri.split("/")[2]
    key = "/".join(s3_uri.split("/")[3:])

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        s3_client.download_file(bucket, key, temp_file.name)
        return temp_file.name

def input_fn(request_body, request_content_type):
    if request_content_type == "application/json":
        input_data = json.loads(request_body)
        s3_uri = input_data['video_path']
        local_path = download_from_s3(s3_uri)
        return {"video_path": local_path}
    raise ValueError(f"Unsupported content type: {request_content_type}")

def output_fn(prediction, response_content_type):
    if response_content_type == "application/json":
        return json.dumps(prediction)
    raise ValueError(f"Unsupported content type: {response_content_type}")

def model_fn(model_dir):
    if not install_ffmpeg():
        raise RuntimeError("FFmpeg installation failed - required for inference")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalSentimentModel().to(device)

    model_path = os.path.join(model_dir, 'model.pth')
    if not os.path.exists(model_path):
        model_path = os.path.join(model_dir, "model", 'model.pth')
        if not os.path.exists(model_path):
            raise FileNotFoundError("Model file not found in path " + model_path)

    logger.info("Loading model from path: %s", model_path)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    return {
        'model': model,
        'tokenizer': AutoTokenizer.from_pretrained('bert-base-uncased'),
        'transcriber': whisper.load_model("base", device="cpu" if device.type == "cpu" else device),
        'device': device
    }

def predict_fn(input_data, model_dict):
    model = model_dict['model']
    tokenizer = model_dict['tokenizer']
    device = model_dict['device']
    video_path = input_data['video_path']

    logger.debug("Starting transcription for video: %s", video_path)
    try:
        result = model_dict['transcriber'].transcribe(video_path, word_timestamps=True)
        logger.debug("Transcription result: %s segments", len(result.get("segments", [])))
        for i, segment in enumerate(result.get("segments", [])):
            logger.debug("Segment %d: start=%s, end=%s, text='%s'", i, segment.get("start"), segment.get("end"), segment.get("text"))
    except Exception as e:
        logger.error("Transcription failed: %s", e)
        result = {"segments": []}

    if not result.get("segments"):
        logger.warning("No segments found in transcription, using dummy segment")
        duration = get_video_duration(video_path)
        result["segments"] = [{"start": 0.0, "end": duration, "text": "No transcription available"}]

    utterance_processor = VideoUtteranceProcessor()
    predictions = []

    for segment in result["segments"]:
        segment_path = None
        try:
            logger.debug("Processing segment: start=%s, end=%s, text='%s'", segment["start"], segment["end"], segment["text"])
            segment_path = utterance_processor.extract_segment(
                video_path, segment["start"], segment["end"])
            video_frames = utterance_processor.video_processor.process_video(segment_path)
            
            try:
                audio_features = utterance_processor.audio_processor.extract_features(segment_path)
            except Exception as e:
                logger.warning("Audio processing failed: %s, using dummy audio features", e)
                audio_features = torch.zeros(1, 64, 300).to(device)

            text_inputs = tokenizer(
                segment["text"], padding="max_length", truncation=True, max_length=128, return_tensors="pt")

            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            video_frames = video_frames.unsqueeze(0).to(device)
            audio_features = audio_features.unsqueeze(0).to(device)

            with torch.inference_mode():
                outputs = model(text_inputs, video_frames, audio_features)
                emotion_probs = torch.softmax(outputs["emotions"], dim=1)[0]
                sentiment_probs = torch.softmax(outputs["sentiments"], dim=1)[0]

                emotion_values, emotion_indices = torch.topk(emotion_probs, 7)
                sentiment_values, sentiment_indices = torch.topk(sentiment_probs, 3)

            predictions.append({
                "start_time": segment["start"],
                "end_time": segment["end"],
                "text": segment["text"],
                "emotions": [
                    {"label": EMOTION_MAP[idx.item()], "confidence": conf.item()}
                    for idx, conf in zip(emotion_indices, emotion_values)
                ],
                "sentiments": [
                    {"label": SENTIMENT_MAP[idx.item()], "confidence": conf.item()}
                    for idx, conf in zip(sentiment_indices, sentiment_values)
                ]
            })
            logger.debug("Segment processed successfully")

        except Exception as e:
            logger.error("Segment processing failed: %s", e)
            continue

        finally:
            if segment_path and os.path.exists(segment_path):
                try:
                    os.remove(segment_path)
                except Exception as e:
                    logger.warning("Failed to clean up segment file: %s", e)

    if not predictions:
        logger.error("No valid segments processed")
        raise ValueError("No valid segments processed")

    logger.debug("Predictions generated: %s utterances", len(predictions))
    return {"utterances": predictions}

def process_local_video(video_path, model_dir="model_normalized"):
    model_dict = model_fn(model_dir)
    input_data = {'video_path': video_path}
    predictions = predict_fn(input_data, model_dict)

    for utterance in predictions["utterances"]:
        logger.info("\nUtterance:")
        logger.info("Start: %ss, End: %ss", utterance['start_time'], utterance['end_time'])
        logger.info("Text: %s", utterance['text'])
        logger.info("\n Top Emotions:")
        for emotion in utterance['emotions']:
            logger.info("%s: %.2f", emotion['label'], emotion['confidence'])
        logger.info("\n Top Sentiments:")
        for sentiment in utterance['sentiments']:
            logger.info("%s: %.2f", sentiment['label'], sentiment['confidence'])
        logger.info("-"*50)

if __name__ == "__main__":
    process_local_video("./dia2_utt3.mp4")