import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm


def compute_hamming_distance(query_logits, train_codes, args, knn_k=200):
    """Compute Hamming distances and perform kNN predictions."""
    dists = torch.cdist(torch.sigmoid(query_logits.to(args.device)), train_codes.to(args.device), p=1).cpu()
    yd, yi = dists.topk(knn_k, dim=1, largest=False, sorted=True)
    return yi


def knn_predict(features, train_features, train_labels, num_classes, args, knn_k=200, knn_t=0.1):
    """kNN Classifier."""
    retrieval_one_hot = torch.zeros(knn_k).to(args.device)
    dist = torch.mm(features.to(args.device), train_features.t().to(args.device))
    bs = features.size(0)
    yd, yi = dist.topk(knn_k, dim=1, largest=True, sorted=True)
    candidates = train_labels.view(1, -1).expand(bs, -1)
    retrieval = torch.gather(candidates.cpu(), 1, yi.cpu())
    retrieval_one_hot.resize_(bs * knn_k, num_classes).zero_()
    retrieval_one_hot.scatter_(1, retrieval.view(-1, 1).to(retrieval_one_hot.device), 1)
    yd_transform = yd.clone().div_(knn_t).exp_()
    probs = torch.sum(torch.mul(retrieval_one_hot.view(bs, -1, num_classes).to(args.device),
                                yd_transform.view(bs, -1, 1).to(args.device)), 1)
    _, predictions = probs.sort(1, True)
    return predictions


def compute_map_at_k(retrieval_indices, retrieval_labels, query_labels, topk=None):
    """Compute Mean Average Precision at k."""
    num_queries = retrieval_indices.size(0)
    APs = []
    for i in range(num_queries):
        gt_label = query_labels[i].item()
        if topk is None:
            retrieved_indices = retrieval_indices[i]
        else:
            retrieved_indices = retrieval_indices[i, :topk]
        retrieved_labels = retrieval_labels[retrieved_indices]
        relevant = (retrieved_labels == gt_label).float()
        num_relevant = relevant.sum()

        if num_relevant == 0:
            APs.append(0.0)
            continue

        precision_at_k = relevant.cumsum(dim=0) / torch.arange(1, topk + 1, device=retrieved_indices.device).float()
        AP = (precision_at_k * relevant).sum() / num_relevant
        APs.append(AP.item())

    return 100 * sum(APs) / num_queries


@torch.no_grad()
def extract_features(encoder, input_processor, data_loader, args):
    """Extract features using the encoder."""

    dataset_sr = {
        "esc50": 44100,
        "gtzan": 22050,
        "speechcommands": 16000,
        "vocalsound": 16000,
        "cremad": 16000,
    }

    original_sr = dataset_sr.get(args.dataset.lower()) 
    if args.model == "clap":
        target_sr = 48000
    elif args.model == "whisper":
        target_sr = 16000
    else:
        target_sr = input_processor.sampling_rate

    resampler = None
    if original_sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=original_sr, new_freq=target_sr)#.to(args.device)

    features, labels = [], []
    for audios, audio_labels in tqdm(data_loader, desc="Extracting Features", total=len(data_loader)):

        if resampler is not None:
            audios = torch.stack([resampler(a.to(torch.float32)) for a in audios])

        if args.model == 'clap':
            audios = input_processor(audios=[a.numpy() for a in audios], sampling_rate=target_sr, return_tensors="pt")
            audios = {name: tensor.to(args.device) for name, tensor in audios.items()}
            feat = encoder(audios['input_features'])
        elif args.model == 'whisper':
            audios = input_processor(audio=[a.numpy() for a in audios], sampling_rate=target_sr, return_tensors="pt")
            audios = {name: tensor.to(args.device) for name, tensor in audios.items()}
            feat = encoder(audios["input_features"])
        else:
            audios = input_processor(audios, sampling_rate=target_sr, return_tensors="pt")
            audios = audios['input_values'].to(args.device)
            if args.model == 'data2vec':
                audios = audios.squeeze()
            feat = encoder(audios)

        features.append(feat)
        labels.append(audio_labels)
    return torch.cat(features, dim=0), torch.cat(labels, dim=0)


@torch.no_grad()
def extract_features_nopca(encoder, input_processor, data_loader, args):
    """Extract features using the encoder."""

    dataset_sr = {
        "esc50": 44100,
        "gtzan": 22050,
        "speechcommands": 16000,
        "vocalsound": 16000,
        "cremad": 16000,
    }

    original_sr = dataset_sr.get(args.dataset.lower()) 
    if args.model == "clap":
        target_sr = 48000
    elif args.model == "whisper":
        target_sr = 16000
    else:
        target_sr = input_processor.sampling_rate

    resampler = None
    if original_sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=original_sr, new_freq=target_sr)#.to(args.device)

    features, labels = [], []
    for audios, audio_labels in tqdm(data_loader, desc="Extracting Features", total=len(data_loader)):

        if resampler is not None:
            audios = torch.stack([resampler(a.to(torch.float32)) for a in audios])

        if args.model == 'clap':
            audios = input_processor(audios=[a.numpy() for a in audios], sampling_rate=target_sr, return_tensors="pt")
            audios = {name: tensor.to(args.device) for name, tensor in audios.items()}
            feat = encoder.get_audio_features(**audios)
        elif args.model == 'whisper':
            audios = input_processor(audio=[a.numpy() for a in audios], sampling_rate=target_sr, return_tensors="pt")
            audios = {name: tensor.to(args.device) for name, tensor in audios.items()}
            outputs = encoder.get_encoder()(**audios)
            feat = outputs['last_hidden_state'].mean(dim=1)
        elif args.model in ['dasheng', 'ced']:
            audios = input_processor(audios, sampling_rate=target_sr, return_tensors="pt")
            audios = {name: tensor.to(args.device) for name, tensor in audios.items()}
            outputs = encoder(**audios)
            feat = outputs.logits
        elif args.model == 'data2vec':
            audios = input_processor(audios, sampling_rate=target_sr, return_tensors="pt")
            audios = audios['input_values'].to(args.device)
            outputs = encoder(audios.squeeze())
            feat = outputs['extract_features'].mean(dim=1)

        features.append(feat)
        labels.append(audio_labels)
    return torch.cat(features, dim=0), torch.cat(labels, dim=0)


def retrieval_knn_eval(encoder, buckethead, input_processor, train_loader, test_loader, args):
    """Perform retrieval and kNN evaluation."""
    encoder.eval()
    buckethead.eval()

    if args.dataset == 'esc50':
        num_classes = 50
    elif args.dataset == 'speechcommands':
        num_classes = 35
    elif args.dataset == 'gtzan':
        num_classes = 10
    elif args.dataset == 'vocalsound':
        num_classes = 6
    elif args.dataset == 'cremad':
        num_classes = 6
    # elif args.dataset == 'fma':
    #     num_classes = 8
    else:
        raise ValueError(f"Unknown number of classes for: {args.dataset}")
    
    knn_k, knn_t = args.knn_k, args.knn_t

    train_features, train_labels = extract_features(encoder, input_processor, train_loader, args)
    test_features, test_labels = extract_features(encoder, input_processor, test_loader, args)

    train_dataset = TensorDataset(train_features)
    train_dataloader = DataLoader(train_dataset, batch_size=args.bs, shuffle=False)
    test_dataset = TensorDataset(test_features, test_labels)
    test_dataloader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False)

    train_logits = []
    for features in train_dataloader:
        features = features[0].to(args.device)
        with torch.no_grad():
            train_logit = buckethead(features).cpu()
        train_logits.append(train_logit)

    train_logits = torch.cat(train_logits, dim=0)
    threshold = 0
    train_codes = (train_logits > threshold).to(dtype=train_features.dtype)

    retrieval_continuous, retrieval_asymhamming, query_labels = [], [], []
    total_num, total_top1, total_asymhamming_top1 = 0, 0., 0.

    for features, labels in tqdm(test_dataloader, desc="Evaluation", total=len(test_dataloader)):
        features, labels = features.to(args.device), labels
        total_num += features.size(0)
        query_labels.append(labels)

        with torch.no_grad():
            test_logit = buckethead(features).cpu()

        test_code = (test_logit > threshold).to(dtype=features.dtype)
        features, test_code, test_logit = features.cpu(), test_code.cpu(), test_logit.cpu()

        dist = torch.mm(F.normalize(features, dim=-1).to(args.device), F.normalize(train_features.t().to(args.device), dim=-1))
        _, topk_idx = dist.topk(train_features.size(0), largest=True, sorted=True)
        retrieval_continuous.append(topk_idx.cpu())

        asymhamming_dist = torch.cdist(torch.sigmoid(test_logit).to(args.device), train_codes.to(args.device), p=1)
        _, topk_idx = asymhamming_dist.topk(train_codes.size(0), largest=False, sorted=True)
        retrieval_asymhamming.append(topk_idx.cpu())

        pred_labels = knn_predict(F.normalize(features, dim=-1), F.normalize(train_features, dim=-1), train_labels, num_classes, args, knn_k, knn_t)
        total_top1 += (pred_labels[:, 0].cpu() == labels).float().sum().item()

        asymhamming_pred = knn_predict(test_logit, train_codes, train_labels, num_classes, args, knn_k)
        total_asymhamming_top1 += (asymhamming_pred[:, 0].cpu() == labels).float().sum().item()

    acc = 100 * total_top1 / total_num
    asymhamming_acc = 100 * total_asymhamming_top1 / total_num

    retrieval_continuous = torch.cat(retrieval_continuous, dim=0)
    retrieval_asymhamming = torch.cat(retrieval_asymhamming, dim=0)
    query_labels_cat = torch.cat(query_labels, dim=0)

    topk = retrieval_continuous.size(1)
    map_cont = compute_map_at_k(retrieval_continuous, train_labels.cpu(), query_labels_cat, topk=topk)
    map_asymhamming = compute_map_at_k(retrieval_asymhamming, train_labels.cpu(), query_labels_cat, topk=topk)

    return {
        'map_cont': map_cont,
        'map_asymhamming': map_asymhamming,
        'acc': acc,
        'asymhamming_acc': asymhamming_acc,
    }


def retrieval_knn_eval_nopca(encoder, input_processor, train_loader, test_loader, args):
    """Perform retrieval and kNN evaluation."""
    encoder.eval()

    if args.dataset == 'esc50':
        num_classes = 50
    elif args.dataset == 'speechcommands':
        num_classes = 35
    elif args.dataset == 'gtzan':
        num_classes = 10
    elif args.dataset == 'vocalsound':
        num_classes = 6
    elif args.dataset == 'cremad':
        num_classes = 6
    # elif args.dataset == 'fma':
    #     num_classes = 8
    else:
        raise ValueError(f"Unknown number of classes for: {args.dataset}")
    
    knn_k, knn_t = args.knn_k, args.knn_t

    train_features, train_labels = extract_features_nopca(encoder, input_processor, train_loader, args)
    test_features, test_labels = extract_features_nopca(encoder, input_processor, test_loader, args)

    test_dataset = TensorDataset(test_features, test_labels)
    test_dataloader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False)

    retrieval_continuous, query_labels = [], []
    total_num, total_top1 = 0, 0.

    for features, labels in tqdm(test_dataloader, desc="Evaluation", total=len(test_dataloader)):
        features, labels = features.to(args.device), labels
        total_num += features.size(0)
        query_labels.append(labels)

        features = features.cpu()

        dist = torch.mm(F.normalize(features).to(args.device), F.normalize(train_features.t().to(args.device), dim=-1))
        _, topk_idx = dist.topk(train_features.size(0), largest=True, sorted=True)
        retrieval_continuous.append(topk_idx.cpu())

        pred_labels = knn_predict(F.normalize(features, dim=-1), F.normalize(train_features, dim=-1), train_labels, num_classes, args, knn_k, knn_t)
        total_top1 += (pred_labels[:, 0].cpu() == labels).float().sum().item()

    acc = 100 * total_top1 / total_num

    retrieval_continuous = torch.cat(retrieval_continuous, dim=0)
    query_labels_cat = torch.cat(query_labels, dim=0)

    topk = retrieval_continuous.size(1)
    map_cont = compute_map_at_k(retrieval_continuous, train_labels.cpu(), query_labels_cat, topk=topk)

    return {
        'map_cont': map_cont,
        'acc': acc,
    }