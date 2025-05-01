import os

from models import MultimodalSentimentModel

# Disable TensorFlow's oneDNN optimizations
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

def count_parameters(model):
    """Counts the number of trainable parameters in a model.
    Args:
        model: The model to count parameters for.
    Returns:
        The number of trainable parameters in the model.
    """
    params_dict = {
        "text_encoder": 0,
        "video_encoder": 0,
        "audio_encoder": 0,
        "fusion_layer": 0,
        "emotion_classifier": 0,
        "sentiment_classifier": 0,
    }

    total_params = 0

    # Iterate through the model's named parameters
    # - Count total No. of parameters in the model
    for name, param in model.named_parameters():

        if param.requires_grad:
            param_count = param.numel()
            total_params += param.numel()

            if "text_encoder" in name:
                params_dict["text_encoder"] += param_count
            elif "video_encoder" in name:
                params_dict["video_encoder"] += param_count
            elif "audio_encoder" in name:
                params_dict["audio_encoder"] += param_count
            elif "fusion_layer" in name:
                params_dict["fusion_layer"] += param_count
            elif "emotion_classifier" in name:
                params_dict["emotion_classifier"] += param_count
            elif "sentiment_classifier" in name:
                params_dict["sentiment_classifier"] += param_count

    return params_dict, total_params


if __name__ == "__main__":
    model = MultimodalSentimentModel()
    params_dict, total_params = count_parameters(model)

    print("=============================================")
    print("Model Parameters Count")
    print("==============================================")
    print("Parameters Breakdown:")
    print("=============================================")
    for component, count in params_dict.items():
        print(f"{component:20s}: {count:,} parameters")
    print("=============================================")
    print("Total Parameters: ", total_params)
    print("=============================================")
    
    # print("=============================================")
    # print("Total Parameters: ", total_params)
    # print("=============================================")
    # print("Parameters Breakdown:")
    # print("=============================================")
    # print("Text Encoder Parameters: ", params_dict["text_encoder"])
    # print("Video Encoder Parameters: ", params_dict["video_encoder"])
    # print("Audio Encoder Parameters: ", params_dict["audio_encoder"])
    # print("Fusion Layer Parameters: ", params_dict["fusion_layer"])
    # print("Emotion Classifier Parameters: ", params_dict["emotion_classifier"])
    # print("Sentiment Classifier Parameters: ", params_dict["sentiment_classifier"])
    # print("=============================================")
