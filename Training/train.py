import os
import sys
import json
import torch
import argparse
import torchaudio
from tqdm import tqdm

from install_ffmpeg import install_ffmpeg
from meld_dataset_1 import prepare_dataloaders
from models import MultimodalSentimentAnalysisModel, MultimodalTrainer

# AWS SageMaker
SM_MODEL_DIR = os.environ.get("SM_MODEL_DIR", ".")
SM_CHANNEL_TRAINING = os.environ.get(
    "SM_CHANNEL_TRAINING", "/opt/ml/input/data/training")
SM_CHANNEL_VALIDATION = os.environ.get(
    "SM_CHANNEL_VALIDATION", "/opt/ml/input/data/validation")
SM_CHANNEL_TEST = os.environ.get(
    "SM_CHANNEL_TEST", "/opt/ml/input/data/test")

# PyTorch CUDA Alloc Config -- Memory Management --
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def parse_args():
    """Parses command line arguments for training a multimodal sentiment analysis model.
    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    # ArgumentParser object
    parser = argparse.ArgumentParser()

    # Model Hyperparameters -- [epochs, batch size, learning rate] --
    parser.add_argument("--epochs", type = int, default = 20, help="Number of epochs to train the model")
    parser.add_argument("--batch_size", type = int, default = 16, help="Batch size for training")
    parser.add_argument("--learning_rate", type = float, default = 0.001, help="Learning rate for the model")

    # Data Directories -- [train, validation, test] --
    parser.add_argument("--train_dir", type = str, default = SM_CHANNEL_TRAINING, help="Directory for training data")
    parser.add_argument("--val_dir", type = str, default = SM_CHANNEL_VALIDATION, help="Directory for validation data")
    parser.add_argument("--test_dir", type = str, default = SM_CHANNEL_TEST, help="Directory for test data")

    # Model Directory
    parser.add_argument("--model_dir", type = str, default = SM_MODEL_DIR, help="Directory to save the model")

    return parser.parse_args()

def main():

    # Install FFMPEG -- [Audio Processing] --
    print("=============================================")
    print("Installing FFMPEG...")
    print("=============================================")
    if not install_ffmpeg():
        print("Error: FFMPEG installation failed. -- Exiting...!")
        sys.exit(1)
    print("=============================================")
    print("FFMPEG installation completed successfully.")
    print("=============================================")
    
    # # Check FFMPEG version
    # print("=============================================")
    # print("FFMPEG version:")
    # print("=============================================")
    # result = os.popen("ffmpeg -version").read()
    # print(result)
    # print("=============================================")
    # # Check FFMPEG installation
    # print("=============================================")
    # print("FFMPEG installation check:")
    # print("=============================================")
    # result = os.popen("which ffmpeg").read()
    # if result:
    #     print("FFMPEG is installed at: ", result.strip())
    # else:
    #     print("FFMPEG is not installed.")
    # print("=============================================")

    print("Available Audio Backends: ", torchaudio.list_audio_backends())
    # print(str(torchaudio.list_audio_backends()))
    print("=============================================")

    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Track CUDA Memory Usage
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        memory_used = torch.cuda.max_memory_allocated() / 1024**3
        print("=============================================")
        print("CUDA Memory Usage")
        print("=============================================")
        print(f"CUDA Device: {torch.cuda.get_device_name(device)}")
        # print(f"CUDA Device Count: {torch.cuda.device_count()}")
        # print(f"CUDA Device Index: {torch.cuda.current_device()}")
        # print(f"CUDA Device Properties: {torch.cuda.get_device_properties(device)}")
        # print(f"CUDA Memory Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        # print(f"CUDA Memory Cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        # print(f"CUDA Memory Peak Allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
        # print(f"CUDA Memory Peak Cached: {torch.cuda.max_memory_reserved() / 1024**3:.2f} GB")
        # print(f"CUDA Memory Allocated (Current): {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        # print(f"CUDA Memory Cached (Current): {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        print(f"CUDA Memory Used: {memory_used:.2f} GB")
        print("=============================================")


    # Data Loaders -- [train, validation, test] --
    train_loder, val_loader, test_loader = prepare_dataloaders(
        # Train Data -- [train_csv, train_video_dir] --
        train_csv = os.path.join(args.train_dir, 'train_sent_emo.csv'),
        train_video_dir = os.path.join(args.train_dir, 'train_splits'),
        # Validation Data -- [dev_csv, dev_video_dir] --
        dev_csv = os.path.join(args.val_dir, 'dev_sent_emo.csv'),
        dev_video_dir = os.path.join(args.val_dir, 'dev_splits_complete'),
        # Test Data -- [test_csv, test_video_dir] --
        test_csv = os.path.join(args.test_dir, 'test_sent_emo.csv'),
        test_video_dir = os.path.join(args.test_dir, 'output_repeated_splits_test'),
        # Batch Size -- [batch_size] --
        batch_size = args.batch_size
    )

    # --- Debugging Information ---
    # Print Data Loader Information
    print("=============================================")
    print("Data Loader Information")
    print("=============================================")
    print(f"Train Data Loader: {train_loder}")
    print(f"Validation Data Loader: {val_loader}")
    print(f"Test Data Loader: {test_loader}")
    print("=============================================")
    # Print Data Loader Lengths
    print("=============================================")
    print("Data Loader Lengths")
    print("=============================================")
    print(f"Train Data Loader Length: {len(train_loder)}")
    print(f"Validation Data Loader Length: {len(val_loader)}")
    print(f"Test Data Loader Length: {len(test_loader)}")
    print("=============================================")
    # Print Data Loader Batch Sizes
    print("=============================================")
    print("Data Loader Batch Sizes")
    print("=============================================")
    print(f"Train Data Loader Batch Size: {train_loder.batch_size}")
    print(f"Validation Data Loader Batch Size: {val_loader.batch_size}")
    print(f"Test Data Loader Batch Size: {test_loader.batch_size}")
    print("=============================================")
    # Print Data Loader Shapes
    print("=============================================")
    print("Data Loader Shapes")
    print("=============================================")
    for i, (video, audio, text, label) in enumerate(train_loder):
        print(f"Batch {i + 1}:")
        print(f"Video Shape: {video.shape}")
        print(f"Audio Shape: {audio.shape}")
        print(f"Text Shape: {text.shape}")
        print(f"Label Shape: {label.shape}")
        break
    print("=============================================")
    # Print Data Loader Sample
    print("=============================================")
    print("Data Loader Sample")
    print("=============================================")
    for i, (video, audio, text, label) in enumerate(train_loder):
        print(f"Batch {i + 1}:")
        print(f"Video Sample: {video[0]}")
        print(f"Audio Sample: {audio[0]}")
        print(f"Text Sample: {text[0]}")
        print(f"Label Sample: {label[0]}")
        break
    print("=============================================")

    print(f"""Training CSV file path: {os.path.join(args.train_dir, 'train_sent_emo.csv')}""")
    print(f"""Training Video Directory: {os.path.join(args.train_dir, 'train_splits')}""")

    # --- Model Training ---
    model = MultimodalSentimentAnalysisModel().to(device)
    trainer = MultimodalTrainer(model, train_loder, val_loader)

    best_val_loss = float("inf")

    metrics_data = {
        "train_losses": [],
        "val_losses": [],
        "epochs": []
    }

    for epoch in tqdm(range(args.epochs), desc="Training Epochs"):
        # --- Training ---
        train_loss = trainer.train_epoch(epoch)
        val_loss, val_metrics = trainer.evaluate(val_loader)

        # --- Track Metrics ---
        metrics_data["train_losses"].append(train_loss["total"])
        metrics_data["val_losses"].append(val_loss["total"])
        metrics_data["epochs"].append(epoch)

        # --- Logging --- in SageMaker format
        print(json.dumps({
            "metrics": [
                {"MetricName": "Training loss", 
                 "Value": train_loss["total"]},
                {"MetricName": "Validation loss", 
                 "Value": val_loss["total"]},
                {"MetricName": "Validation: emotion precision", 
                 "Value": val_metrics["emotion_precision"]},
                {"MetricName": "validation: emotion accuracy", 
                 "Value": val_metrics["emotion_accuracy"]},
                {"MetricName": "validation: sentiment precision", 
                 "Value": val_metrics["sentiment_precision"]},
                {"MetricName": "validation: sentiment accuracy", 
                 "Value": val_metrics["sentiment_accuracy"]}
            ]
        }))

        # Track CUDA Memory Usage
        if torch.cuda.is_available():
            memory_used = torch.cuda.max_memory_allocated() / 1024**3
            print("=============================================")
            print("CUDA Memory Usage [GPU] -- [Peak] --")
            print("=============================================")
            print(f"CUDA Device: {torch.cuda.get_device_name(device)}")
            print(f"Peak GPU Memory Used: {memory_used:.2f} GB")
            print("=============================================")

        # Save the best Model -- [Model Directory] --
        print("=============================================")
        print("Saving Best Model...")
        print("=============================================")
        
        if val_loss["total"] < best_val_loss:
            best_val_loss = val_loss["total"]
            print("=============================================")
            print("Best Validation Loss: ", best_val_loss)
            print("=============================================")
            print("Saving Model...")
            print("=============================================")
            # Save the model state dict
            torch.save(model.state_dict(), os.path.join(args.model_dir, "model.pth"))
            print("=============================================")
            print("Best Model Saved")
            print("=============================================")

    # After Training -- Completed --
    # Evaluate the model on the test set
    print("=============================================")
    print("Evaluating Model on Test Set...")
    print("=============================================")
    test_loss, test_metrics = trainer.evaluate(test_loader, phase="test")
    
    metrics_data["test_loss"] = test_loss["total"]

    # --- Logging --- in SageMaker format
    print(json.dumps({
        "metrics": [
            {"MetricName": "Testing loss", 
                "Value": test_loss["total"]},
            {"MetricName": "Testing: emotion precision", 
             "Value": test_metrics["emotion_precision"]},
            {"MetricName": "Testing: emotion accuracy", 
             "Value": test_metrics["emotion_accuracy"]},
            {"MetricName": "Testing: sentiment precision", 
             "Value": test_metrics["sentiment_precision"]},
            {"MetricName": "Testing: sentiment accuracy", 
             "Value": test_metrics["sentiment_accuracy"]},
        ]
    }))


    
    # print("=============================================")
    # print("Training Multimodal Sentiment Analysis Model")
    # print("=============================================")
    # print("Loading Arguments...")
    # print("=============================================")
    # args = pargs_args()
    # print("Arguments Loaded")
    # print("=============================================")
    # print("Arguments:")
    # print("=============================================")
    # print(f"Epochs: {args.epochs}")
    # print(f"Batch Size: {args.batch_size}")
    # print(f"Learning Rate: {args.learning_rate}")
    # print(f"Training Directory: {args.train_dir}")
    # print(f"Validation Directory: {args.val_dir}")
    # print(f"Test Directory: {args.test_dir}")
    # print(f"Model Directory: {args.model_dir}")
    # print("=============================================")
    # print("Starting Training...")
    # print("=============================================")

if __name__ == "__main__":
    main()