import torch
import torchaudio
import torch.nn as nn
from tqdm import tqdm


def compute_pca(model, input_processor, train_loader, args):
    """
    Compute PCA on features extracted from the audio model.
    Args:
        model: The neural network audio model.
        input_processor: Feature extractor for the audio data.
        train_loader: DataLoader for the training audio data.
        args: Additional arguments including device and bits.
    Returns:
        U, S, V: Matrices from PCA.
    """
    # Dataset-specific sample rates (defaults)
    dataset_sr = {
        "esc50": 44100,
        "gtzan": 22050,
        "speechcommands": 16000,
        "vocalsound": 16000,
        "cremad": 16000,
        # "fma": 16000,
    }

    original_sr = dataset_sr.get(args.dataset.lower()) 
    if args.model == "clap":
        target_sr = 48000
    elif args.model == "whisper":
        target_sr = 16000
    else:
        target_sr = input_processor.sampling_rate

    # Pre-create resampler only if needed
    resampler = None
    if original_sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=original_sr, new_freq=target_sr)

    features_list = []
    model.eval()

    for batch_idx, (audios, _) in enumerate(tqdm(train_loader, desc="PCA", total=len(train_loader))):
        with torch.no_grad():

            if resampler is not None:
                audios = torch.stack([resampler(a.to(torch.float32)) for a in audios])

            if args.model in ['dasheng', 'ced']:
                audios = input_processor(audios, sampling_rate=target_sr, return_tensors="pt")
                audios = {name: tensor.to(args.device) for name, tensor in audios.items()}
                outputs = model(**audios)
                # print(outputs['logits'].shape, outputs['hidden_states'].shape) # [256, 768], [256, 125, 768]
                features = outputs.logits
                # features = outputs['hidden_states'].mean(dim=1)

            elif args.model == 'data2vec':
                audios = input_processor(audios, sampling_rate=target_sr, return_tensors="pt")
                audios = audios['input_values'].to(args.device)
                outputs = model(audios.squeeze())
                # print(outputs['extract_features'].shape, outputs['last_hidden_state'].shape) # [32, 249, 512], [32, 249, 768]
                # features = outputs.last_hidden_state.mean(dim=1)
                features = outputs['extract_features'].mean(dim=1)

            elif args.model == 'clap':
                audios = input_processor(audios=[a.numpy() for a in audios], sampling_rate=target_sr, return_tensors="pt")
                audios = {name: tensor.to(args.device) for name, tensor in audios.items()}
                features = model.get_audio_features(**audios)

            elif args.model == 'whisper':
                audios = input_processor(audio=[a.numpy() for a in audios], sampling_rate=target_sr, return_tensors="pt")
                audios = {name: tensor.to(args.device) for name, tensor in audios.items()}
                outputs = model.get_encoder()(**audios)
                features = outputs['last_hidden_state'].mean(dim=1)

            else:
                raise ValueError(f"Unknown model type: {args.model}")

            features_list.append(features.cpu())

    features = torch.cat(features_list, dim=0)
    mean_features = features.mean(dim=0, keepdim=True)
    U, S, V = torch.pca_lowrank(features, q=args.bits, center=True, niter=2)
    return U, S, V, mean_features


class AudioEncoderWithPCA(nn.Module):
    """
    An audio model followed by a PCA projection layer.
    """
    def __init__(self, model, input_processor, input_dim, pca_matrix, mean_features, args):
        """
        Initialize the audio model with PCA projection.
        Args:
            model: The neural network audio model.
            input_processor: Feature extractor for the audio data.
            input_dim: Dimension of the input features.
            pca_matrix: The PCA matrix V.
            args: Additional arguments including device and bits.
        """
        super().__init__()
        self.model = model
        self.input_processor = input_processor
        self.projection = nn.Linear(input_dim, args.bits, bias=False)
        self.projection.weight.data = pca_matrix[:, :args.bits].T
        self.projection = self.projection.to(args.device)
        self.mean_features = mean_features.to(args.device)
        self.args = args

    @torch.no_grad()
    def forward(self, x):
        """
        Forward pass through the audio model and PCA projection.
        Args:
            x: Input audio tensor.
        Returns:
            Projected features.
        """
        # Extract features using the feature extractor

        if self.args.model in ['dasheng', 'ced']:        

            outputs = self.model(x)
            x_features = outputs.logits
            # x_features = outputs['hidden_states'].mean(dim=1)

        elif self.args.model == 'data2vec':

            outputs = self.model(x)
            x_features = outputs['extract_features'].mean(dim=1)
            # x_features = outputs.last_hidden_state.mean(dim=1)

        elif self.args.model == 'clap':

            x_features = self.model.get_audio_features(x)

        elif self.args.model == 'whisper':

            outputs = self.model.get_encoder()(x)
            x_features = outputs['last_hidden_state'].mean(dim=1)

        else:
            raise ValueError(f"Unknown model type: {self.args.model}")
        
        # Features centering
        x_features = x_features - self.mean_features

        # Project using PCA
        return self.projection(x_features)