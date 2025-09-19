from transformers import AutoModel, AutoModelForAudioClassification, AutoFeatureExtractor, AutoProcessor, AutoModelForSpeechSeq2Seq, ClapModel, ClapProcessor, ClapFeatureExtractor
import torch.nn as nn


def get_audio_encoder(args):
    """Load and initialize an audio model based on the simplified model name."""
    # Mapping from simplified names to full model names
    model_mapping = {
        "dasheng": "mispeech/dasheng-base",
        "data2vec": "facebook/data2vec-audio-base",
        "whisper": "openai/whisper-base",
        "clap": "laion/clap-htsat-unfused",
        "ced": "mispeech/ced-base"
    }

    # Get the full model name from the mapping
    full_model_name = model_mapping.get(args.model)
    if not full_model_name:
        raise ValueError("Unsupported simplified model name.")

    if "dasheng" in full_model_name:
        component = AutoFeatureExtractor.from_pretrained(full_model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(full_model_name, trust_remote_code=True)

    elif "ced" in full_model_name:
        component = AutoFeatureExtractor.from_pretrained(full_model_name, trust_remote_code=True)
        model = AutoModelForAudioClassification.from_pretrained(full_model_name, trust_remote_code=True)

    elif "data2vec" in full_model_name:
        component = AutoProcessor.from_pretrained(full_model_name)
        model = AutoModel.from_pretrained(full_model_name)

    elif "whisper" in full_model_name:
        component = AutoProcessor.from_pretrained(full_model_name)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(full_model_name)

    elif "clap" in full_model_name:
        component = ClapProcessor.from_pretrained(full_model_name)
        #component = ClapFeatureExtractor.from_pretrained(full_model_name)
        model = ClapModel.from_pretrained(full_model_name)

    print(f"{full_model_name} model loaded successfully.")
    
    model = model.to(args.device)
    model.eval()

    if full_model_name in ["mispeech/dasheng-base", "openai/whisper-base", "mispeech/ced-base"]:
        dim = 768
    elif full_model_name in ["facebook/data2vec-audio-base", "laion/clap-htsat-unfused", "openai/whisper-base"]:
        dim = 512

    return model, component, dim


def get_buckethead(bits):
    """Initialize an orthogonal linear layer without bias."""
    buckethead = nn.Linear(bits, bits, bias=False)
    nn.init.orthogonal_(buckethead.weight)
    return buckethead