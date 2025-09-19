import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm


def compute_map_at_k(retrieval_indices, retrieval_labels, query_labels, topk):
    """Compute Mean Average Precision at k (supports single-label & multi-label)."""
    num_queries = retrieval_indices.size(0)
    APs = []

    is_multilabel = (query_labels.dim() == 2)  # [N, C] means multi-label

    for i in range(num_queries):
        retrieved_indices = retrieval_indices[i, :topk]
        retrieved_labels = retrieval_labels[retrieved_indices]

        if is_multilabel:
            gt_label = query_labels[i].float()
            relevant = (retrieved_labels * gt_label).sum(dim=1) > 0
        else:
            gt_label = query_labels[i].item()
            relevant = (retrieved_labels == gt_label)

        relevant = relevant.float()
        num_relevant = relevant.sum()

        if num_relevant == 0:
            APs.append(0.0)
            continue

        precision_at_k = relevant.cumsum(dim=0) / torch.arange(1, topk + 1, device=retrieved_indices.device).float()
        AP = (precision_at_k * relevant).sum() / num_relevant
        APs.append(AP.item())

    return 100 * sum(APs) / num_queries


@torch.no_grad()
def extract_features(encoder, data_loader, args, desc):
    """Extract features using the encoder."""
    features, labels = [], []
    for images, image_labels in tqdm(data_loader, desc=desc):
        images = images.to(args.device)
        if args.nopca:
            if args.model == "dfn":
                feat = encoder.get_image_embeddings(images).cpu()
            else:
                output = encoder(images)
                if isinstance(output, dict) and 'x_norm_clstoken' in output:
                    feat = output['x_norm_clstoken'].cpu()
                else:
                    raise ValueError("Encoder output does not contain 'x_norm_clstoken'.")
        else:
            feat = encoder(images).cpu()
        features.append(feat)
        labels.append(image_labels)
    return torch.cat(features, dim=0), torch.cat(labels, dim=0)


def retrieval_eval(encoder, buckethead, database_loader, query_loader, args):
    """Perform retrieval evaluation."""
    encoder.eval()
    buckethead.eval()

    if args.dataset == 'cifar10':
        map_k = 1000
    elif args.dataset in ['flickr25k', 'coco', 'nuswide']:
        map_k = 5000
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    database_features, database_labels = extract_features(encoder, database_loader, args, desc="Extracting database features")
    query_features, query_labels = extract_features(encoder, query_loader, args, desc="Extracting query features")

    database_dataset = TensorDataset(database_features)
    database_dataloader = DataLoader(database_dataset, batch_size=args.bs, shuffle=False)
    query_dataset = TensorDataset(query_features, query_labels)
    query_dataloader = DataLoader(query_dataset, batch_size=args.bs, shuffle=False)

    # binarize database codes
    database_logits = []
    for (features,) in database_dataloader:
        features = features.to(args.device)
        with torch.no_grad():
            database_logit = buckethead(features).cpu()
        database_logits.append(database_logit)
    database_logits = torch.cat(database_logits, dim=0)
    database_codes = (database_logits > 0).to(dtype=database_features.dtype)

    retrieval_continuous, retrieval_asymhamming, query_labels = [], [], []
    total_num = 0

    for features, labels in tqdm(query_dataloader, desc="Evaluation"):
        features = features.to(args.device)
        total_num += features.size(0)
        query_labels.append(labels)

        with torch.no_grad():
            query_logit = buckethead(features).cpu()

        query_code = (query_logit > 0).to(dtype=features.dtype)
        features, query_code = features.cpu(), query_code.cpu()

        # retrieval indices
        dist = torch.mm(F.normalize(features, dim=-1), F.normalize(database_features, dim=-1).t())
        _, topk_idx = dist.topk(map_k, largest=True, sorted=True)
        retrieval_continuous.append(topk_idx.cpu())

        asymhamming_dist = torch.cdist(torch.sigmoid(query_logit), database_codes, p=1)
        _, topk_idx = asymhamming_dist.topk(map_k, largest=False, sorted=True)
        retrieval_asymhamming.append(topk_idx.cpu())

    retrieval_continuous = torch.cat(retrieval_continuous, dim=0)
    retrieval_asymhamming = torch.cat(retrieval_asymhamming, dim=0)
    query_labels_cat = torch.cat(query_labels, dim=0)

    map_cont = compute_map_at_k(retrieval_continuous, database_labels.cpu(), query_labels_cat, topk=map_k)
    map_asymhamming = compute_map_at_k(retrieval_asymhamming, database_labels.cpu(), query_labels_cat, topk=map_k)

    return {'map_cont': map_cont, 'map_asymhamming': map_asymhamming,}


def retrieval_eval_nopca(encoder, database_loader, query_loader, args):
    """Perform retrieval evaluation (no PCA)."""
    encoder.eval()

    if args.dataset == 'cifar10':
        map_k = 1000
    elif args.dataset in ['flickr25k', 'coco', 'nuswide']:
        map_k = 5000
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    database_features, database_labels = extract_features(encoder, database_loader, args, desc="Extracting database features")
    query_features, query_labels = extract_features(encoder, query_loader, args, desc="Extracting query features")

    query_dataset = TensorDataset(query_features, query_labels)
    query_dataloader = DataLoader(query_dataset, batch_size=args.bs, shuffle=False)

    retrieval_continuous, query_labels = [], []
    total_num = 0

    for features, labels in tqdm(query_dataloader, desc="Evaluation"):
        features = features.to(args.device)
        total_num += features.size(0)
        query_labels.append(labels)

        features = features.cpu()

        dist = torch.mm(F.normalize(features), F.normalize(database_features, dim=-1).t())
        _, topk_idx = dist.topk(map_k, largest=True, sorted=True)
        retrieval_continuous.append(topk_idx.cpu())

    retrieval_continuous = torch.cat(retrieval_continuous, dim=0)
    query_labels_cat = torch.cat(query_labels, dim=0)

    map_cont = compute_map_at_k(retrieval_continuous, database_labels.cpu(), query_labels_cat, topk=map_k)

    return {'map_cont': map_cont,}