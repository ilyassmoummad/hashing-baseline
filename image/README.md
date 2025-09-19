# Image Retrieval Hashing

This folder contains code to perform **image retrieval hashing** using pre-trained Vision Transformer (ViT) models and PCA-based hashing with random orthogonal projection.  
The pipeline generates binary hash codes of configurable dimensions (e.g., 16, 32, or 64 bits) for efficient image retrieval.

---

## Datasets

Supported datasets:
- **CIFAR-10** – Automatically downloaded via PyTorch.
- **Flickr25K, COCO, NUS-WIDE** – Require manual download.

### Download Instructions
#### COCO 2014
```bash
wget --no-check-certificate -c https://images.cocodataset.org/zips/train2014.zip && unzip -q train2014.zip
wget --no-check-certificate -c https://images.cocodataset.org/zips/val2014.zip && unzip -q val2014.zip
wget --no-check-certificate -c https://images.cocodataset.org/annotations/annotations_trainval2014.zip && unzip -q annotations_trainval2014.zip
```

#### FLICKR25K 
```bash
wget http://press.liacs.nl/mirflickr/mirflickr25k.v3b/mirflickr25k.zip && unzip mirflickr25k.zip
```

#### NUS-WIDE
To download the [NUS-WIDE dataset](https://www.kaggle.com/datasets/ssaysham/nuswide-for-deephash), you’ll need to install and configure the Kaggle CLI:

1. Go to your [Kaggle account settings](https://www.kaggle.com/account).
2. Scroll down to the **API** section and click **Create New API Token**.  
   This will download a file called `kaggle.json`.
3. Move the file to the correct location and set permissions:

   ```bash
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```
4. Download the dataset using Kaggle CLI:

    ```bash
    kaggle datasets download ssaysham/nuswide-for-deephash
    ```

5. Unzip the dataset:

    ```bash
    unzip nuswide-for-deephash.zip -d nuswide-for-deephash
    ```

## Pretrained Models

The code supports ViT backbones trained with [DINOv2](https://arxiv.org/abs/2304.07193), [SimDINOv2](https://arxiv.org/abs/2502.10385), and [DFN](https://arxiv.org/abs/2309.17425). DFN is automatically downloaded via Huggingface:

- **DFN (DFN-2B)**:  
  [https://huggingface.co/apple/DFN2B-CLIP-ViT-B-16](https://huggingface.co/apple/DFN2B-CLIP-ViT-B-16)

For DINOv2 and SimDINOv2, download the following model weights:

- **DINOv2 with registers (LVD-142)**:  
  [https://github.com/facebookresearch/dinov2](https://github.com/facebookresearch/dinov2)

- **SimDINOv2 (ImageNet-1k)**:  
  [https://github.com/RobinWu218/SimDINO](https://github.com/RobinWu218/SimDINO)

---

## Usage

### Basic Command
```bash
python3 main.py \
  --model path_to_dinov2_or_simdinov2_weights.pth or 'dfn' \
  --data_dir path_to_dataset \
  --dataset cifar10|flickr25k|coco|nuswide \
  --n_workers 4 \
  --bits 16 \
  --bs 256 \
```

### Arguments
| Argument            | Description                                                                    |
|---------------------|--------------------------------------------------------------------------------|
| `--model`           | Path to downloaded model weights (for Sim/DINOv2) or 'dfn' for DFN model.      |
| `--data_dir`        | Path to dataset folder.                                                        |
| `--dataset`         | Dataset name (`cifar10`, `flickr25k`, `coco`, or `nuswide`).                   |
| `--n_workers`       | Number of workers for data loading (default: 4).                               |
| `--bits`            | Number of bits for hashing (`16`, `32`, or `64`).                              |
| `--bs`              | Batch size for processing (default: 256).                                      |
| `--nopca`           | (Optional) Disable PCA-based hashing (retrieval with original model features). |
| `--save`            | (Optional) Save results to text files.                                         |
| `--results_dir`     | (Optional) Path to directory for saving results.                               |
| `--n_runs`          | Number of evaluation runs (default: 10).                                       |

## Example Runs

**CIFAR10 without hashing using DFN**
```bash
python3 main.py --model dfn --data_dir path_to_cifar10 --dataset cifar10 --nopca --bs 256
```

**COCO with 64-bit hashing using DINOv2**
```bash
python3 main.py --model path_to_dinov2weights.pth --data_dir path_to_coco --dataset coco --bits 64 --bs 256 --n_run 10
```

**NUS-WIDE with 32-bit hashing using SimDINOv2**
```bash
python3 main.py --model path_to_simdinov2weights.pth --data_dir path_to_nuswide --dataset nuswide --bits 32 --bs 256 --n_run 10
```

**FLICKR25k with 16-bit hashing using DFN**
```bash
python3 main.py --model dfn.pth --data_dir path_to_flickr25k --dataset flickr25k --bits 16 --bs 256 --n_run 10
```

**Notes**
- PCA + random projection is applied by default for binary hashing.  
- Use `--nopca` to disable PCA and use raw model features instead.