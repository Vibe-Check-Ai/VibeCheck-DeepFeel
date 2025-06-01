import os
import torch
import torch.nn as nn

from datetime import datetime
from transformers import BertModel
from torchvision import models as vision_models
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_score, accuracy_score

from meld_dataset_1 import MELDDataset


### =========================== Encoders =========================== ###
# 1. Text Encoder: BERT Model
# 2. Video Encoder: 3D ResNet-18 Model
# 3. Audio Encoder: Convolutional Neural Network (CNN) Model
### ================================================================ ###

# Text Encoder: BERT Model
# - Input: Textual Data (Tokenized)
# - Output: Textual Features (128-Dimensional)
# - Pre-trained BERT Model: bert-base-uncased
class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")

        # Freeze the BERT model parameters
        for param in self.bert.parameters():
            param.requires_grad = False

        # Projection layer: reduce output dimension from 768 -> 128
        self.projection = nn.Linear(768, 128)

    def forward(self, input_ids, attention_mask):
        # Get the BERT model outputs (BERT embeddings)
        outputs = self.bert(input_ids = input_ids, attention_mask = attention_mask)

        # Use [CLS] Token Representation
        pooler_output = outputs.pooler_output

        # Apply projection layer -> reduce the dimensionality
        return self.projection(pooler_output)
    

# Video Encoder: 3D ResNet-18 Model
# - Input: Video Data (3D Tensors)
# - Output: Video Features (128-Dimensional)
# - Pre-trained 3D ResNet-18 Model: r3d_18
class VideoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = vision_models.video.r3d_18(pretrained = True)

        # Freeze the backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False

        # No. of features -> (3D ResNet-18)
        num_features = self.backbone.fc.in_features

        # Backbone Model: 3D ResNet-18 -- Layers -- (1,2) => Replacement Process
        # 1. Remove the last fully connected layer (fc)
        # 2. Add a new fully connected layer (fc) with 128 output features
        # 3. Add ReLU activation function
        # 4. Add Dropout layer -- Dropout probability = 0.2 --
        # 5. Set the new fully connected layer as the backbone's fc layer
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        # [batch_size, num_frames, channels, height, width] -> [batch_size, num_frames, 128]
        # (B, T, C, H, W) -> (B, C, T, H, W)
        x = x.transpose(1, 2)

        return self.backbone(x)


# Audio Encoder: Convolutional Neural Network (CNN) Model
# - Input: Audio Data (1D Tensors)
# - Output: Audio Features (128-Dimensional)
# - Pre-trained CNN Model: Custom CNN Model
class AudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            # Lower Level Features
            # - 1st Convolutional Layer (1D)
            # - 1st Batch Normalization Layer
            # - 1st ReLU Activation Function
            # - Max Pooling Layer
            nn.Conv1d(64, 64, kernel_size = 3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            # Higher Level Features
            # - 2nd Convolutional Layer (1D)
            # - 2nd Batch Normalization Layer
            # - 2nd ReLU Activation Function
            # - Adaptive Average Pooling Layer
            nn.Conv1d(64, 128, kernel_size = 3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        # Freeze the convolutional layers parameters
        for param in self.conv_layers.parameters():
            param.requires_grad = False

        # Projection Layer: -- Trainable --
        # - Linear Layer (128 -> 128)
        # - ReLU Activation Function
        # - Dropout Layer -- Dropout Probability = 0.2 --
        self.projection = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        # Remove the channel dimension (1D)
        # [batch_size, 1, 64, 300] -> [batch_size, 64, 300]
        # (B, C, T, F) -> (B, T, F)
        x = x.squeeze(1)

        # Convolutional Layers: Extract Features
        # - features output: [batch_size, 128, 1]
        features = self.conv_layers(x)

        # Projection layer
        return self.projection(features.squeeze(-1))
### ================================================================ ###


### ========================= Fusion Layer ========================= ###
# - Input:
# -- Textual Features (128-Dimensional)
# -- Video Features (128-Dimensional)
# -- Audio Features (128-Dimensional)
# - Output:
# -->>> Combined Features (384-Dimensional)
# - Fusion Method: Concatenation
class MultimodalSentimentModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoders
        self.text_encoder = TextEncoder()
        self.video_encoder = VideoEncoder()
        self.audio_encoder = AudioEncoder()

        # Fusion Layer:
        # - Concatenate the features from all three encoders
        # - Linear Layer: Reduce the dimensionality from 384 -> 256
        # - Batch Normalization Layer
        # - ReLU Activation Function
        # - Dropout Layer -- Dropout Probability = 0.3 --
        # - Output: Combined Features (256-Dimensional)
        self.fusion_layer = nn.Sequential(
            nn.Linear(128 * 3, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Classification Layer heads:
        # 1) Emotion Classification Head
        # - Linear Layer: Reduce the dimensionality from 256 -> 64
        # - ReLU Activation Function
        # - Dropout Layer -- Dropout Probability = 0.2 --
        # - Linear Layer: Output 7 Classes (Emotions)
        self.emotion_classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 7) # [Anger, Disgust, Sadness, Joy, Neutral, Surprise and Fear]
        )

        # 2) Sentiment Classification Head
        # - Linear Layer: Reduce the dimensionality from 256 -> 64
        # - ReLU Activation Function
        # - Dropout Layer -- Dropout Probability = 0.2 --
        # - Linear Layer: Output 3 Classes (Sentiments)
        # - Output: Combined Features (3 Classes)
        self.sentiment_classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 3) # [Positive, Negative, Neutral]
        )

    def forward(self, text_inputs, video_frames, audio_features):
        # Text Encoder: Get the textual features
        # - Input: Tokenized Textual Data (input_ids, attention_mask)
        # - Output: Textual Features (128-Dimensional)
        text_features = self.text_encoder(
            text_inputs["input_ids"],
            text_inputs["attention_mask"]
        )

        # Video Encoder: Get the video features
        # - Input: Video Data (3D Tensors)
        # - Output: Video Features (128-Dimensional)
        video_features = self.video_encoder(video_frames)

        # Audio Encoder: Get the audio features
        # - Input: Audio Data (1D Tensors)
        # - Output: Audio Features (128-Dimensional)
        audio_features = self.audio_encoder(audio_features)

        # Concatenate the features from all three encoders
        # - Combined Features: [batch_size, 128 * 3 = 384] -- [index 0, index 1]
        combined_features = torch.cat(
            [
                text_features,
                video_features,
                audio_features
            ], dim = 1 # Dimension (1) -> index 1 for features!
        )

        # Apply the fusion layer to the combined features
        # - Output: Fused Features (256-Dimensional)
        # - Fused Features: [batch_size, 256] -- [index 0, index 1]
        fused_features = self.fusion_layer(combined_features)

        # Emotion Classification
        # - Input: Fused Features (256-Dimensional)
        # - Output: Emotion Classification (7 Classes)
        # - Emotion Classification: [batch_size, 7] -- [index 0, index 1]
        emotion_output = self.emotion_classifier(fused_features)

        # Sentiment Classification
        # - Input: Fused Features (256-Dimensional)
        # - Output: Sentiment Classification (3 Classes)
        # - Sentiment Classification: [batch_size, 3] -- [index 0, index 1]
        sentiment_output = self.sentiment_classifier(fused_features)

        return {
            "emotions": emotion_output,
            "sentiments": sentiment_output
        }
### ================================================================ ###
# Compute Class Weights
# - Input: MELD Dataset
# - Output:
# -- Emotion Weights: Tensor of size 7
# -- Sentiment Weights: Tensor of size 3
def compute_class_weights(dataset):
    # Compute class distributions for emotions and sentiments
    # - Emotion Labels: 7 Classes (Anger, Disgust, Sadness, Joy, Neutral, Surprise and Fear)
    # - Sentiment Labels: 3 Classes (Positive, Negative, Neutral)
    # - Emotion Counts: Tensor of size 7
    # - Sentiment Counts: Tensor of size 3
    # - Skipped Samples: 0
    # - Total Samples: Length of the dataset
    emotion_counts = torch.zeros(7)
    sentiment_counts = torch.zeros(3)
    skipped = 0
    total = len(dataset)

    print("\Counting class distributions...")
    for i in range(total):
        sample = dataset[i]

        if sample is None:
            skipped += 1
            continue

        emotion_label = sample['emotion_label']
        sentiment_label = sample['sentiment_label']

        emotion_counts[emotion_label] += 1
        sentiment_counts[sentiment_label] += 1

    # Total Samples: Length of the dataset
    # Valid Samples: Total Samples - Skipped Samples
    valid = total - skipped
    print(f"Skipped samples: {skipped}/{total}")

    # Print class distributions
    print("\nClass distribution")
    print("Emotions:")
    emotion_map = {0: 'anger', 1: 'disgust', 2: 'fear',
                   3: 'joy', 4: 'neutral', 5: 'sadness', 6: 'surprise'}
    for i, count in enumerate(emotion_counts):
        print(f"{emotion_map[i]}: {count/valid:.2f}")

    print("\nSentiments:")
    sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
    for i, count in enumerate(sentiment_counts):
        print(f"{sentiment_map[i]}: {count/valid:.2f}")

    # Calculate class weights for emotions and sentiments
    # - Emotion Weights: 1.0 / Emotion Counts
    # - Sentiment Weights: 1.0 / Sentiment Counts
    emotion_weights = 1.0 / emotion_counts
    sentiment_weights = 1.0 / sentiment_counts

    # Normalize weights
    # - Emotion Weights: Emotion Weights / Sum of Emotion Weights
    # - Sentiment Weights: Sentiment Weights / Sum of Sentiment Weights
    emotion_weights = emotion_weights / emotion_weights.sum()
    sentiment_weights = sentiment_weights / sentiment_weights.sum()

    return emotion_weights, sentiment_weights

### ====================== Multimodal Trainer ====================== ###
# - Input:
# -- Model: Multimodal Sentiment Model
# -- Train DataLoader: Training Dataset
# -- Validation DataLoader: Validation Dataset
# - Output:
# -- Train the model for one epoch
# -- Validate the model on the validation set
# -- Return the average loss and evaluation metrics
# - Evaluation Metrics:
# Emotion & Sentiment [precision + accuracy]
class MultimodalTrainer:
    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Dataset: MELD Dataset -- Information --
        # - Train Dataset
        # - Validation Dataset
        train_size = len(train_loader.dataset)
        val_size = len(val_loader.dataset)

        # Print the dataset sizes
        print("===================================")
        print("MELD Dataset:")
        print("===================================")
        print("Train Dataset Size:", train_size)
        print("Validation Dataset Size:", val_size)
        print("===================================")
        print("Batches per Epoch:", f"{len(train_loader):,}")
        print("===================================")

        # TensorBoard Logging:
        # - Log Directory: "runs/experiment_name"
        # - Experiment Name: "MELD_Train_Val_Logs"
        # - Current Time: "YYYY-MM-DD_HH-MM-SS"
        # - Summary Writer: TensorBoard Summary Writer
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        if "SM_MODEL_DIR" in os.environ:
            # Linux Systems -- AWS SageMaker --
            base_dir = "/opt/ml/output/tensorboard"
        else:
            # Local Systems -- Windows --
            base_dir = "runs"

        # Log Directory: "runs/experiment_name"
        # log_dir = f"{base_dir}/run_{timestamp}"
        log_dir = os.path.join(base_dir, f"MELD_Train_Val_Logs_{timestamp}")
        self.writer = SummaryWriter(log_dir = log_dir)
        self.global_step = 0

        # Optimizer: Adam Optimizer
        # - Weight Decay: 1e-5 (Regularization Term)
        # - Parameters: Text Encoder, Video Encoder, Audio Encoder, Fusion Layer, Emotion Classifier, Sentiment Classifier
        # Learning Rates [ 1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001 ]:
        # - Text Encoder: 8e-6
        # - Video Encoder: 8e-5
        # - Audio Encoder: 8e-5
        # - Fusion Layer: 5e-4
        # - Emotion Classifier: 5e-4
        # - Sentiment Classifier: 5e-4
        self.optimizer = torch.optim.Adam([
            {"params": model.text_encoder.parameters(), "lr": 8e-6},
            {"params": model.video_encoder.parameters(), "lr": 8e-5},
            {"params": model.audio_encoder.parameters(), "lr": 8e-5},
            {"params": model.fusion_layer.parameters(), "lr": 5e-4},
            {"params": model.emotion_classifier.parameters(), "lr": 5e-4},
            {"params": model.sentiment_classifier.parameters(), "lr": 5e-4},
        ], weight_decay = 1e-5)

        # Learning Rate Scheduler: ReduceLROnPlateau
        # - Mode: "min" (Reduce the learning rate when a metric has stopped improving)
        # - Factor: 0.1 (Reduce the learning rate by a factor of 0.1)
        # - Patience: 2 (Number of epochs with no improvement after which learning rate will be reduced)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode = "min",
            factor = 0.1,
            patience = 2
        )

        self.current_train_loss = None

        # Class Weights:
        # - Emotion Weights: Tensor of size 7
        # - Sentiment Weights: Tensor of size 3
        print("Computing class weights...")
        emotion_weights, sentiment_weights = compute_class_weights(train_loader.dataset)

        # Move the class weights to the same device as the model
        # - Device: GPU/CPU
        print("Moving class weights to the model's device...")
        device = next(model.parameters()).device

        self.emotion_weights = emotion_weights.to(device)
        self.sentiment_weights = sentiment_weights.to(device)

        print(f"Emotion weights on device: {self.emotion_weights.device}")
        print(f"Sentiments weights on device: {self.sentiment_weights.device}")

        # Loss Function: CrossEntropyLoss
        # - Label Smoothing: 0.05 (Regularization Technique)
        # - Emotion Classification: 7 Classes (Anger, Disgust, Sadness, Joy, Neutral, Surprise and Fear)
        # - Sentiment Classification: 3 Classes (Positive, Negative, Neutral)
        self.emtion_criterion = nn.CrossEntropyLoss(
            label_smoothing = 0.05,
            weight = self.emotion_weights
        )
        self.sentiment_criterion = nn.CrossEntropyLoss(
            label_smoothing = 0.05,
            weight = self.sentiment_weights
        )

    def log_metrics(self, loss, metrics = None, phase = "train"):
        # Train Phase
        if phase == "train":
            self.current_train_loss = loss
        # Validation Phase
        else:
            # Loss logs
            # - Train Loss: Current Train Loss
            # - Validation Loss: Validation Loss
            # - Global Step: No. of Epochs
            self.writer.add_scalar("loss/total/train", self.current_train_loss["total"], self.global_step)
            self.writer.add_scalar("loss/total/val", loss["total"], self.global_step)

            # Emotion logs
            self.writer.add_scalar("loss/emotion/train", self.current_train_loss["emotion"], self.global_step)
            self.writer.add_scalar("loss/emotion/val", loss["emotion"], self.global_step)
    
            # Sentiment logs
            self.writer.add_scalar("loss/sentiment/train", self.current_train_loss["sentiment"], self.global_step)
            self.writer.add_scalar("loss/sentiment/val", loss["sentiment"], self.global_step)

        if metrics:
            print("Metrics dictionary contents:", metrics)
            print(f"Logging metrics for {phase}:")
            # Emotion Metrics: -- Precision + Accuracy --
            print(f"{phase}/emotion/precision")
            print(f"{phase}/emotion/accuracy")
            self.writer.add_scalar(f"{phase}/emotion/precision", metrics["emotion_precision"], self.global_step)
            self.writer.add_scalar(f"{phase}/emotion/accuracy", metrics["emotion_accuracy"], self.global_step)
            
            # Sentiment Metrics: -- Precision + Accuracy --
            print(f"{phase}/sentiment/precision")
            print(f"{phase}/sentiment/accuracy")
            self.writer.add_scalar(f"{phase}/sentiment/precision", metrics["sentiment_precision"], self.global_step)
            self.writer.add_scalar(f"{phase}/sentiment/accuracy", metrics["sentiment_accuracy"], self.global_step)
            
            # --- IMPORTANT --- !!!
            # - Flush the writer to ensure all data is written to disk
            self.writer.flush()

    def train_epoch(self):
        # Set the model to training mode
        self.model.train()
        
        # Running loss for the current epoch
        # - Total Loss: 0
        # - Emotion Loss: 0
        # - Sentiment Loss: 0
        running_loss = {"total": 0, "emotion": 0, "sentiment": 0}

        # Training Loop:
        # - Iterate over the training data in batches
        # - Move the data to the device (GPU/CPU)
        # - Zero the gradients
        # - Forward pass: Get the model outputs
        # - Loss Calculation: [Emotion Loss, Sentiment Loss, Total Loss]
        # - Backward pass: Compute gradients
        # - Gradient Clipping: Prevent exploding gradients
        # - Update the model parameters
        # - Track the running loss
        for batch in self.train_loader:
            device = next(self.model.parameters()).device
            text_inputs = {
                "input_ids": batch["text_inputs"]["input_ids"].to(device),
                "attention_mask": batch["text_inputs"]["attention_mask"].to(device)
            }
            video_frames = batch["video_frames"].to(device)
            audio_features = batch["audio_features"].to(device)

            emotion_labels = batch["emotion_label"].to(device)
            sentiment_labels = batch["sentiment_label"].to(device)

            # Zero the gradients
            self.optimizer.zero_grad()

            # Forward pass: Get the model outputs
            outputs = self.model(
                text_inputs,
                video_frames,
                audio_features
            )

            # Loss Calculation:
            # - Emotion Loss: CrossEntropyLoss
            # - Sentiment Loss: CrossEntropyLoss
            # - Total Loss: Emotion Loss + Sentiment Loss
            emotion_loss = self.emtion_criterion(
                outputs["emotions"],
                emotion_labels
            )
            sentiment_loss = self.sentiment_criterion(
                outputs["sentiments"],
                sentiment_labels
            )
            total_loss = emotion_loss + sentiment_loss

            # Backward pass: Compute gradients
            total_loss.backward()

            # Gradient Clipping: Prevent exploding gradients
            # - Clip gradients to a maximum norm of 1.0
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm = 1.0
            )

            # Update the model parameters
            # - Step the optimizer
            self.optimizer.step()

            # Track the running loss
            running_loss["total"] += total_loss.item()
            running_loss["emotion"] += emotion_loss.item()
            running_loss["sentiment"] += sentiment_loss.item()

            # Log -- TensorBoard -- [Train Loss]
            self.log_metrics({
                "total": total_loss.item(),
                "emotion": emotion_loss.item(),
                "sentiment": sentiment_loss.item()
            })

            # Global Step: No. of Epochs
            self.global_step += 1
        
        # Average the running loss
        # - Total Loss: Total Loss / Number of batches
        # - Emotion Loss: Emotion Loss / Number of batches
        # - Sentiment Loss: Sentiment Loss / Number of batches
        # - Return the average loss
        return {
            k: v / len(self.train_loader) for k, v in running_loss.items()
        }

    def evaluate(self, data_loader, phase =  "val"):
        # Set the model to evaluation mode
        # - Disable dropout layer
        self.model.eval()

        # Running loss for the validation set
        # - Total Loss: 0
        # - Emotion Loss: 0
        # - Sentiment Loss: 0
        loss = {"total": 0, "emotion": 0, "sentiment": 0}

        all_emotion_preds = []
        all_emotion_labels = []

        all_sentiment_preds = []
        all_sentiment_labels = []
        
        # Validation Loop:
        # - Iterate over the validation data in batches
        # - Move the data to the device (GPU/CPU)
        # - Forward pass: Get the model outputs
        # - Loss Calculation: [Emotion Loss, Sentiment Loss, Total Loss]
        # - Emotion & Sentiment [Predictions + Labels]
        # - Track the running loss
        with torch.inference_mode():
            for batch in data_loader:
                device = next(self.model.parameters()).device
                text_inputs = {
                    "input_ids": batch["text_inputs"]["input_ids"].to(device),
                    "attention_mask": batch["text_inputs"]["attention_mask"].to(device)
                }
                video_frames = batch["video_frames"].to(device)
                audio_features = batch["audio_features"].to(device)

                emotion_labels = batch["emotion_label"].to(device)
                sentiment_labels = batch["sentiment_label"].to(device)

                # Forward pass: Get the model outputs
                outputs = self.model(
                    text_inputs,
                    video_frames,
                    audio_features
                )

                # Loss Calculation:
                # - Emotion Loss: CrossEntropyLoss
                # - Sentiment Loss: CrossEntropyLoss
                # - Total Loss: Emotion Loss + Sentiment Loss
                emotion_loss = self.emtion_criterion(
                    outputs["emotions"],
                    emotion_labels
                )
                sentiment_loss = self.sentiment_criterion(
                    outputs["sentiments"],
                    sentiment_labels
                )
                total_loss = emotion_loss + sentiment_loss

                # Emtion Predictions + Labels:
                # - Emotion Predictions: Argmax of the model outputs
                # - Emotion Labels: True Labels
                # - Append the predictions and labels to the lists
                # - Convert to numpy arrays for evaluation
                # - Move to CPU for evaluation
                all_emotion_preds.extend(outputs["emotions"].argmax(dim = 1).cpu().numpy())
                all_emotion_labels.extend(emotion_labels.cpu().numpy())

                # Sentiment Predictions
                # - Sentiment Predictions: Argmax of the model outputs
                # - Sentiment Labels: True Labels
                # - Append the predictions and labels to the lists
                # - Convert to numpy arrays for evaluation
                # - Move to CPU for evaluation
                all_sentiment_preds.extend(outputs["sentiments"].argmax(dim = 1).cpu().numpy())
                all_sentiment_labels.extend(sentiment_labels.cpu().numpy())

                # Track the running loss
                loss["total"] += total_loss.item()
                loss["emotion"] += emotion_loss.item()
                loss["sentiment"] += sentiment_loss.item()


        # Average the running loss
        # - Total Loss: Total Loss / Number of batches
        # - Emotion Loss: Emotion Loss / Number of batches
        # - Sentiment Loss: Sentiment Loss / Number of batches
        # - Return the average loss
        avg_loss = {
            k: v / len(data_loader) for k, v in loss.items()
        }

        # Evalutation Metrics:
        # - Emotion Precision: Weighted Precision Score
        # - Emotion Accuracy: Accuracy Score
        emotion_precision = precision_score(
            all_emotion_labels,
            all_emotion_preds,
            average = "weighted"
        )
        emotion_accuracy = accuracy_score(
            all_emotion_labels,
            all_emotion_preds
        )

        # Sentiment Precision: Weighted Precision Score
        # - Sentiment Accuracy: Accuracy Score
        sentiment_precision = precision_score(
            all_sentiment_labels,
            all_sentiment_preds,
            average = "weighted"
        )
        sentiment_accuracy = accuracy_score(
            all_sentiment_labels,
            all_sentiment_preds
        )

        # Log -- TensorBoard -- [Validation Loss + Metrics(precision + accuracy)]
        self.log_metrics(avg_loss,{
            "emotion_precision": emotion_precision,
            "emotion_accuracy": emotion_accuracy,
            "sentiment_precision": sentiment_precision,
            "sentiment_accuracy": sentiment_accuracy
        }, phase = phase)
        
        # Only step -- Phase == "Validation" --
        if phase == "val":
            # Scheduler Step: -- with Avg Loss --
            self.scheduler.step(avg_loss["total"])

        # return avg_loss, {
        #     "emotion": {
        #         "precision": emotion_precision,
        #         "accuracy": emotion_accuracy
        #     },
        #     "sentiment": {
        #         "precision": sentiment_precision,
        #         "accuracy": sentiment_accuracy
        #     }
        # }

        # Reurn [Avg, Metrics]
        # - Avg Loss: Average Loss
        # - Metrics(Emotion): Emotion Precision, Emotion Accuracy, 
        # - Metrics(Sentiment): Sentiment Precision, Sentiment Accuracy
        return avg_loss, {
            "emotion_precision": emotion_precision,
            "emotion_accuracy": emotion_accuracy,
            "sentiment_precision": sentiment_precision,
            "sentiment_accuracy": sentiment_accuracy
        }
### ================================================================ ###

if __name__ == "__main__":
    dataset = MELDDataset(
        "../Dataset/MELD.Raw/train/train_sent_emo.csv",
        "../Dataset/MELD.Raw/train/train_splits"
    )

    sample = dataset[0]
    MultimodalSentimentModel = MultimodalSentimentModel()
    MultimodalSentimentModel.eval()

    text_inputs = {
        "input_ids": sample["text_inputs"]["input_ids"].unsqueeze(0),
        "attention_mask": sample["text_inputs"]["attention_mask"].unsqueeze(0)
    }
    video_frames = sample["video_frames"].unsqueeze(0)
    audio_features = sample["audio_features"].unsqueeze(0)

    with torch.inference_mode():
        outputs = MultimodalSentimentModel(
            text_inputs,
            video_frames,
            audio_features
        )

        emotions_probs = torch.softmax(outputs["emotions"], dim = 1)[0]
        sentiments_probs = torch.softmax(outputs["sentiments"], dim = 1)[0]

        #  Mapping: Anger, Disgust, Sadness, Joy, Neutral, Surprise and Fear
        emotion_map = {
            0: "anger",
            1: "disgust",
            2: "fear",
            3: "joy",
            4: "neutral",
            5: "sadness",
            6: "surprise"
        }

        #  Mapping: Positive, Negative, Neutral
        sentiment_map = {
            0: "negative",
            1: "neutral",
            2: "positive"
        }
    
    print("Predictions for utterance (1 Sample Instance):")
    
    print("Emotion Predictions:")
    for i, prob in enumerate(emotions_probs):
        print(f"Emotion: {emotion_map[i]} - Probability: {prob:.4f}")

    print("\nSentiment Predictions:")

    for i, prob in enumerate(sentiments_probs):
        print(f"Sentiment: {sentiment_map[i]} - Probability: {prob:.4f}")


    






    
"""     
    batch_size = 2

    x = torch.randn(batch_size, 1, 64, 300)  # Dummy input for audio encoder 
    x_squeezed = x.squeeze(1)  # Remove the channel dimension (1D)
    
    print("Audio Encoder Output Shape:", (x).shape)
    print("Squeezed Input Shape:", x_squeezed.shape)




  
    # Test the model
    text_encoder = TextEncoder()
    video_encoder = VideoEncoder()
    audio_encoder = AudioEncoder()

    # Dummy input for testing
    input_ids = torch.randint(0, 1000, (2, 10))
    attention_mask = torch.ones((2, 10))
    video_input = torch.randn(2, 16, 3, 224, 224)
    audio_input = torch.randn(2, 1, 128)

    text_output = text_encoder(input_ids, attention_mask)
    video_output = video_encoder(video_input)
    audio_output = audio_encoder(audio_input)

    print("Text Encoder Output Shape:", text_output.shape)
    print("Video Encoder Output Shape:", video_output.shape)
    print("Audio Encoder Output Shape:", audio_output.shape)
""" 