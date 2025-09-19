# Hashing-Baseline: Rethinking Hashing in the Age of Pretrained Models

[![arXiv](https://img.shields.io/badge/arXiv-2509.14427-b31b1b.svg)](https://arxiv.org/abs/2509.14427)

## 👩‍💻 Authors

<sup>1</sup> Ilyass Moummad, <sup>1,2</sup> Kawtar Zaher, <sup>3</sup> Lukas Rauch, <sup>1</sup> Alexis Joly

<sup>1</sup> INRIA, LIRMM, Université de Montpellier, France <br>
<sup>2</sup> Institut National de l’Audiovisuel, France <br>
<sup>3</sup> University of Kassel, Germany 

---

## 🏗️ Repository Overview

This repository demonstrates a **simple, training-free baseline** for retrieval of **images** and **audio**. It works by **combining deep pre-trained features with traditional hashing steps** — **PCA**, **random orthogonal projection**, and a **thresholding operation** — to produce compact binary codes.

The resulting **Hashing-Baseline** produces compact binary codes that:  
- Retain semantic similarity from the pretrained features  
- Allow fast similarity search  

This setup provides a straightforward reference for comparing more complex or learned hashing methods on large datasets.

---

## ⚙️ Method Overview

<img src="figures/Hashing-Baseline.png" alt="Overview of Hashing-Baseline" width="800"/>

The **Hashing-Baseline** pipeline consists of four main steps:

1. **Feature Extraction**  
   Extract high-dimensional features from images or audio using pre-trained models.

2. **Dimensionality Reduction with PCA**  
   Apply Principal Component Analysis (PCA) to reduce feature dimensionality while retaining key semantic information.

3. **Random Orthogonal Projection and Binarization**  
   Project PCA-reduced features with a random orthogonal matrix and binarize by checking whether values are **greater than 0**, resulting in compact 16-bit hash codes.

4. **Retrieval**  
   Perform similarity search in **Hamming space** using the binary codes.

This simple pipeline shows that even standard hashing techniques can preserve semantic similarity when applied on strong pretrained features, without any additional training.

---

## 🔎 Retrieval Example

The figure below shows **two query images** from the **Flickr25K dataset** and their **5 nearest neighbors**, retrieved using features from a **SimDINO ViT-B/16 model pretrained on ImageNet-1K (100 epochs)**.

- **Original features** → continuous feature vectors extracted from the backbone (CLS token).  
- **Hashed 16-bit codes** → binary codes produced by applying our Hashing-Baseline to the original features.  

<p align="center">
  <img src="figures/5NN_retrieval.png" alt="5-NN Retrieval" width="800"/>
</p>

This demonstrates that **Hashing-Baseline preserves semantic similarity**, even with compact 16-bit codes.

## 📂 Repository Structure

- **`image/`** — Contains code for image retrieval hashing.  
  Supports datasets: `cifar10`, `flickr25k`, and `coco`.  
  Supported models: `dfn` and DINOv2/SimDINOv2 via their checkpoints (`.pth`).  
  See [image/README.md](./image/README.md) for detailed instructions.

- **`audio/`** — Contains code for audio retrieval hashing.  
  Supports datasets: `esc50`, `gtzan`, and `speechcommands`.  
  Supports models: `clap`, `ced`, and `dasheng`.  
  See [audio/README.md](./audio/README.md) for detailed instructions.

---

## 🚀 Getting Started

1. **Clone the repository**  
   ```bash
   git clone https://github.com/ilyassmoummad/hashing-baseline
   cd hashing-baseline
   ```
2. **Create a Python environment**
   ```bash
   python -m venv envhashingbaseline
   source envhashingbaseline/bin/activate
   python -m pip install -r requirements.txt
   ```
4. Choose your modality folder (`image` or `audio`).
5. Follow the instructions in the README file of the respective folder to download and set up the datasets.
6. Run the provided scripts with your desired model, dataset, and hashing parameters.

---

## 📚 References

#### Image Models
- DFN: https://arxiv.org/abs/2309.17425
- DINOv2: https://arxiv.org/abs/2304.07193  
- SimDINOv2: https://arxiv.org/abs/2502.10385  
#### Audio Models
- CLAP: https://arxiv.org/abs/2211.06687  
- CED: https://arxiv.org/abs/2308.11957  
- Dasheng: https://arxiv.org/abs/2406.06992

---

## 📝 To cite this work:

```
@misc{hashingbaseline,
      title={Hashing-Baseline: Rethinking Hashing in the Age of Pretrained Models}, 
      author={Ilyass Moummad and Kawtar Zaher and Lukas Rauch and Alexis Joly},
      year={2025},
      eprint={2509.14427},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2509.14427}, 
}
```