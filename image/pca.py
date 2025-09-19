import torch
import torch.nn as nn
from tqdm import tqdm


def compute_pca(encoder, train_loader, args):
    """
    Compute PCA on features extracted from the encoder.

    Args:
        encoder: The neural network encoder.
        train_loader: DataLoader for the training data.
        args: Additional arguments including device and bits.

    Returns:
        U, S, V: Matrices from PCA.
    """
    # List to store extracted features
    features_list = []

    # Set the encoder to evaluation mode
    encoder.eval()

    # Extract features from the training data
    for batch_idx, (images, _) in enumerate(tqdm(train_loader, desc="PCA")):
        with torch.no_grad():
            if args.model == "dfn":
                features = encoder.get_image_embeddings(images.to(args.device))
            else:
                output = encoder(images)
                if isinstance(output, dict) and 'x_norm_clstoken' in output:
                    features = output['x_norm_clstoken']
                else:
                    raise ValueError("Encoder output does not contain 'x_norm_clstoken'.")
            features_list.append(features.cpu())

    # Concatenate all features
    features = torch.cat(features_list, dim=0)

    # mean features
    mean_features = features.mean(dim=0, keepdim=True)

    # Perform PCA
    U, S, V = torch.pca_lowrank(features, q=args.bits, center=True, niter=2)

    return U, S, V, mean_features


class ImageEncoderWithPCA(nn.Module):
    """
    A neural network encoder followed by a PCA projection layer.
    """

    def __init__(self, encoder, input_dim, pca_matrix, mean_features, args):
        """
        Initialize the encoder with PCA projection.

        Args:
            encoder: The neural network encoder.
            input_dim: Dimension of the input features.
            pca_matrix: The PCA matrix V.
            args: Additional arguments including device and bits.
        """
        super().__init__()
        self.encoder = encoder

        # Define the projection layer using PCA components
        self.projection = nn.Linear(input_dim, args.bits, bias=False)
        self.projection.weight.data = pca_matrix[:, :args.bits].T
        self.projection = self.projection.to(args.device)
        self.mean_features = mean_features.to(args.device)
        self.args = args

    def forward(self, x):
        """
        Forward pass through the encoder and PCA projection.

        Args:
            x: Input tensor.

        Returns:
            Projected features.
        """
        # Extract the normalized CLS token (ViT)
        if self.args.model == "dfn":
            features = self.encoder.get_image_embeddings(x)
        else:
            output = self.encoder(x)
            if isinstance(output, dict) and 'x_norm_clstoken' in output:
                features = output['x_norm_clstoken']
            else:
                raise ValueError("Encoder output does not contain 'x_norm_clstoken'.")

        # Features centering
        features = features - self.mean_features

        # Project using PCA
        return self.projection(features)