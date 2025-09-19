# Audio Retrieval Hashing

This folder contains code to perform **audio retrieval hashing** using pre-trained audio models and PCA-based hashing with random orthogonal projection.    
The pipeline generates binary hash codes of configurable dimensions (e.g., 16, 32, or 64 bits) for efficient audio retrieval.

---

## Datasets

Supported datasets:
- **ESC-50** – Automatically downloaded via Datasets (Huggingface).
- **GTZAN, VocalSound, CREMA-D** – Require manual download.

### Download Instructions
### GTZAN
```bash
for i in {0..9}; do wget "https://zenodo.org/record/14722472/files/gtzan_fold_${i}_0000000.tar?download=1" -O "gtzan_fold_${i}_0000000.tar"; done
for f in gtzan_fold_*.tar; do dir="${f%.tar}"; mkdir -p "$dir" && tar -xf "$f" -C "$dir"; done
```

### VocalSound
```bash
curl -s "https://zenodo.org/api/records/14650192" | jq -r '.files[].key | "https://zenodo.org/record/14650192/files/\(.)?download=1"' | xargs -n 1 -I {} wget -c --content-disposition "{}"
mkdir train ; mkdir valid ; mkdir test
mv *train*tar train/ ; mv *valid*tar val/ ; mv *test*tar test/
for dir in train val test; do for f in $dir/*.tar; do tar -xf "$f" -C "$dir"; done; done
```

### CREMA-D
```bash
curl -s "https://zenodo.org/api/records/14646870" | jq -r '.files[].key | "https://zenodo.org/record/14646870/files/\(.)?download=1"' | xargs -n 1 -I {} wget -c --content-disposition "{}"
mkdir train ; mkdir valid ; mkdir test
mv *train*tar train/ ; mv *valid*tar val/ ; mv *test*tar test/
for dir in train val test; do for f in $dir/*.tar; do tar -xf "$f" -C "$dir"; done; done
```

## Models

The following pre-trained audio backbones are supported:  
- [**CLAP**](https://arxiv.org/abs/2211.06687) -
  [https://huggingface.co/laion/clap-htsat-unfused](https://huggingface.co/laion/clap-htsat-unfused)

- [**CED**](https://arxiv.org/abs/2308.11957) -
  [https://huggingface.co/mispeech/ced-base](https://huggingface.co/mispeech/ced-base)

- [**Dasheng**](https://arxiv.org/abs/2406.06992) -
  [https://huggingface.co/mispeech/dasheng-base](https://huggingface.co/mispeech/dasheng-base)

---

## Usage

### Basic Command
```bash
python3 main.py \
  --model clap|ced|dasheng \
  --data_dir path_to_dataset_cache \
  --dataset esc50|gtzan|vocalsound|cremad \
  --n_workers 4 \
  --bits 16 \
  --bs 32 \
```

### Arguments
| Argument        | Description                                                                    |
|-----------------|--------------------------------------------------------------------------------|
| `--model`       | Model backbone (`clap`, `ced`, or `dasheng`).                                  |
| `--data_dir`    | Path to dataset cache folder (automatically populated on first run).           |
| `--dataset`     | Dataset name (`esc50`, `gtzan`, `cremad`, or `vocalsound`).                    |
| `--n_workers`   | Number of workers for data loading (default: 4).                               |
| `--bits`        | Number of bits for hashing (`16`, `32`, or `64`).                              |
| `--bs`          | Batch size for processing (default: 32).                                       |
| `--nopca`       | (Optional) Disable PCA-based hashing (retrieval using raw model features).     |
| `--save`        | (Optional) Save results to text files.                                         |
| `--results_dir` | (Optional) Path to directory for saving results.                               |
| `--n_runs`      | Number of evaluation runs (default: 10).                                       |

## Example Runs

**VocalSound (CED model) without hashing**
```bash
python3 main.py --model ced --data_dir path_to_vocalsound --dataset vocalsound --n_workers 4 --bs 256 --nopca
```

**ESC-50 (CLAP model) with 16-bit hashing**
```bash
python3 main.py --model clap --data_dir path_to_esc50 --dataset esc50 --n_workers 4 --bits 16 --bs 32 --n_runs 10
```

**CREMA-D (CLAP model) with 32-bit hashing**
```bash
python3 main.py --model clap --data_dir path_to_cremad --dataset cremad --n_workers 4 --bits 32 --bs 32 --n_runs 10
```

**GTZAN (Dasheng model) with 64-bit hashing**
```bash
python3 main.py --model dasheng --data_dir path_to_gtzan --dataset gtzan --n_workers 4 --bits 64 --bs 256 --n_runs 10
```

**Notes**
- PCA + random projection is applied by default for binary hashing.  
- Use `--nopca` to disable PCA and use raw model features instead.