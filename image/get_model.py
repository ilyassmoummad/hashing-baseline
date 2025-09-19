from models import vit_small, vit_base, vit_large
import open_clip
import torch
import torch.nn as nn
import re


class DFN(nn.Module):
    """DFN multimodal encoder wrapper (image + text) without preprocessing, like SigLIP2Wrapper."""

    def __init__(self, model_type="vitb", device="cuda"):
        super().__init__()
        self.device = device

        if model_type == "vitb":
            ckpt = "hf-hub:apple/DFN2B-CLIP-ViT-B-16"
            self.model, self.preprocess = open_clip.create_model_from_pretrained(ckpt)
            self.tokenizer = open_clip.get_tokenizer("ViT-B-16")
        elif model_type == "vitl":
            ckpt = "hf-hub:apple/DFN2B-CLIP-ViT-L-14-39B"
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(ckpt)
            self.tokenizer = open_clip.get_tokenizer(ckpt)
        else:
            raise ValueError(f"Model type {model_type} not supported. Choose ['vitb','vitl'].")

        self.model.to(device).eval()

    def tokenize_text(self, texts):
        return self.tokenizer(texts, context_length=self.model.context_length).to(self.device)

    def get_image_embeddings(self, pixel_values):
        """pixel_values: [B, 3, H, W] tensor, already preprocessed."""
        pixel_values = pixel_values.to(self.device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            return self.model.encode_image(pixel_values).float()

    def get_text_embeddings(self, input_ids):
        """input_ids: [B, L] tensor, already tokenized."""
        input_ids = input_ids.to(self.device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            return self.model.encode_text(input_ids)


def load_state_dict(weights_path):
    """Load and process the state dictionary for the encoder."""
    # if "SimDINO" in weights_path:
    if "16" in weights_path: # SimDINOv2 use 16 in patch size and DINOv2 use 14
        state_dict = torch.load(weights_path)["teacher"]
        state_dict = {k.removeprefix("backbone."): v for k, v in state_dict.items() if k.startswith("backbone.")}
        state_dict = {remap_key(k): v for k, v in state_dict.items()}
    else:
        state_dict = torch.load(weights_path)
    return state_dict


def remap_key(k):
    """Remap keys in the state dictionary."""
    return re.sub(r"blocks\.\d+\.(?=\d+)", "blocks.", k)


def initialize_encoder(weights_path, device):
    """Initialize the encoder based on the weights path."""
    # img_size = 524 if "SimDINO" not in weights_path else 224
    # patch_size = 16 if "SimDINO" in weights_path else 14

    path_str = weights_path.lower()
    
    if "simdino" in path_str:
        img_size = 224
        patch_size = 16
    elif "dinov2" in path_str:
        if "16" in path_str:
            patch_size = 16
            img_size = 224
        else:
            patch_size = 14
            img_size = 524

    if "vitl" in weights_path:
        encoder = vit_large(
            patch_size=patch_size,
            img_size=img_size,
            init_values=0.1 if "SimDINO" in weights_path else 1e-5,
            block_chunks=0,
            num_register_tokens=4 if 'reg4' in weights_path else 0
        )
    elif "vitb" in weights_path:
        encoder = vit_base(
            patch_size=patch_size,
            img_size=img_size,
            init_values=0.1 if "SimDINO" in weights_path else 1e-5,
            block_chunks=0,
            num_register_tokens=4 if 'reg4' in weights_path else 0
        )
    elif "vits" in weights_path:
        encoder = vit_small(
            patch_size=14,
            img_size=524,
            init_values=1e-5,
            block_chunks=0,
            num_register_tokens=4 if 'reg4' in weights_path else 0
        )
    else:
        raise ValueError("Unsupported model type in weights path.")

    return encoder


def get_image_encoder(args):
    """Get the encoder and its embedding dimension."""

    if args.model.endswith('.pth') is True:
        encoder = initialize_encoder(args.model, args.device)
        state_dict = load_state_dict(args.model)
        encoder.load_state_dict(state_dict, strict=True)
        embed_dim = encoder.embed_dim
    elif args.model == "dfn":
        encoder = DFN(model_type="vitb", device=args.device)
        embed_dim = 512
    else:
        raise ValueError("Unsupported model type or path.")
    encoder = encoder.to(args.device)
    encoder.eval()
    return encoder, embed_dim


def get_buckethead(bits):
    """Initialize an orthogonal linear layer without bias."""
    buckethead = nn.Linear(bits, bits, bias=False)
    nn.init.orthogonal_(buckethead.weight)
    return buckethead