# MVA 2026 — Research Projects & Practical Work

> This repository is a **monorepo of submodules** gathering all projects, practical work, and research-oriented implementations developed during the [MVA (Mathématiques, Vision, Apprentissage)](https://www.master-mva.com/) 2026 program at ENS Paris-Saclay.
>
> It spans a broad range of topics across modern machine learning and deep learning, from generative models and inverse problems to LLM efficiency and GPU kernel engineering.

---

## Table of Contents

1. [Repository Structure](#1-repository-structure)
2. [Submodules Overview](#2-submodules-overview)
   - [Generative Image Modeling](#-generativeimagingmodeling)
   - [Hyperspectral Imaging Unmixing](#-hyperspectral-imaging-unmixing)
   - [Medical Image Analysis](#-medicalimageanalysis)
   - [Representation Learning & Computer Vision](#-representation-learning--computer-vision)
   - [Diffusion Policy](#-diffusionpolicy)
   - [AI Games](#-ai-games)
   - [LLM KV-Cache & LoRA Efficiency](#-llm-kv-cache--lora-efficiency)
   - [Controlled LLM Generation](#-controlled-llm-generation)
   - [GPU LLM Flash Attention](#-gpu-llm-flash-attention)
   - [Timeseries & Graph Learning](#-timeseries--graph-learning)
3. [Getting Started](#3-getting-started)
4. [Topics at a Glance](#4-topics-at-a-glance)

---

## 1. Repository Structure

```
MVA-2026/
│
├── GenerativeImageModeling/              # Guided diffusion for image inverse problems
├── HyperspectralImagingUnmixing/         # Hyperspectral unmixing with PnP priors
├── MedicalImageAnalysis/                 # Medical imaging coursework and practicals
├── Representation-Learning-Computer-Vision/  # SSL, DINO, contrastive learning
├── DiffusionPolicy/                      # Diffusion-based robot policy learning
├── AI-games/                             # AI for Games: MCTS, RL agents
├── LLM-kv-cache-LoRA-efficiency/         # KV-cache compression and LoRA fine-tuning
├── controlled-LLM-generation/            # Constrained / guided text generation
├── gpu_llm_flash-attention/              # CUDA kernels and FlashAttention experiments
├── timeseries-graph-learning/            # Time-series and graph neural networks
│
├── .gitmodules                           # Submodule declarations
└── README.md
```

Each folder is an independent git submodule with its own history, dependencies, and documentation.

---

## 2. Submodules Overview

---

### 🎨 GenerativeImageModeling

> **Guided diffusion for image inverse problems — training-free posterior sampling**

A research implementation of two guidance algorithms that steer a pre-trained DDPM/DDIM prior at inference time to solve degradation problems (inpainting, super-resolution, motion blur, JPEG2000) without any fine-tuning.

**Key methods:**

- **DPS** (Diffusion Posterior Sampling) — gradient of the reconstruction error injected at each reverse step:

$$x_{t-1}^{\text{DPS}} = x_{t-1}^{\text{DDPM}} - \zeta \cdot \frac{\nabla_{x_t} \| \mathcal{H}(\hat{x}_0) - y \|_2^2}{\| \mathcal{H}(\hat{x}_0) - y \|_2}$$

- **PiGDM** (Pseudo-Inverse Guided Diffusion) — Moore-Penrose projection in data space:

$$g_t = \nabla_{x_t} \left[ \langle \text{sg}(v_t), \hat{x}_0 \rangle \right], \quad v_t = \mathcal{H}^\dagger y - \mathcal{H}^\dagger \mathcal{H} \hat{x}_0$$

Both methods exploit Tweedie's formula $\hat{x}_0 = (x_t - \sqrt{1 - \bar{\alpha}_t} \, \varepsilon_\theta) / \sqrt{\bar{\alpha}_t}$ to estimate the clean image at each step.

**Stack:** PyTorch, OpenAI guided-diffusion UNet (FFHQ 256), DDPM/DDIM samplers, FFT-based motion blur, Haar wavelet JPEG2000 operator.

**Papers:** [DPS (Chung et al., ICLR 2023)](https://arxiv.org/abs/2209.14687) · [PiGDM (Song et al., ICLR 2023)](https://arxiv.org/abs/2305.10483) · [DDIM (Song et al., ICLR 2021)](https://arxiv.org/abs/2010.02502)

🔗 [GenerativeImageModeling](https://github.com/KHOUTAIBI/GenerativeImageModeling)

---

### 🌈 Hyperspectral Imaging Unmixing

> **Constrained matrix factorization with Plug-and-Play priors for spectral decomposition**

Hyperspectral images capture tens to hundreds of narrow spectral bands per pixel. Because spatial resolution is limited, each pixel is typically a mixture of several materials. This project tackles **hyperspectral unmixing**: recovering endmember spectra $A$ and abundance maps $S$ from the observed image $X$.

**The Linear Mixing Model:**

$$X = AS + N, \quad A \in \mathbb{R}^{L \times K}, \quad S \in \mathbb{R}^{K \times P}$$

**Physical constraints on abundances:**

$$S \geq 0, \quad \sum_{k=1}^{K} S_{k,p} = 1 \quad \forall p \quad \Longrightarrow \quad S_{\cdot,p} \in \Delta^K$$

**Optimization objective:**

$$\min_{A, S} \frac{1}{2} \| X - AS \|_F^2 + \lambda R(A, S)$$

**Plug-and-Play (PnP) prior:** instead of an explicit $R$, a learned denoiser $D_\sigma$ replaces the proximal step. The **GS-PnP** interpretation gives it variational grounding:

$$D_\sigma(x) = x - \nabla g_\sigma(x)$$

making the overall objective $F(Z) = f(Z) + \lambda g_\sigma(Z)$ fully differentiable and optimizer-compatible.

**Stack:** Python, NumPy, PyTorch (denoiser), Jupyter notebook, synthetic data generation (Dirichlet abundances, dead-leaves spatial structure).

🔗 [HyperspectralImagingUnmixing](https://github.com/KHOUTAIBI/HyperspectralImagingUnmixing)

---

### 🏥 Medical Image Analysis

> **Deep learning applied to clinical imaging pipelines**

Practical work and coursework focused on applying deep learning to medical imaging problems. Topics include segmentation, classification, and analysis of clinical image data (MRI, CT, histology), with an emphasis on practical implementation and evaluation.

**Typical tasks covered:**
- Image segmentation (UNet-style architectures)
- Classification with limited labeled data
- Data augmentation strategies for medical images
- Evaluation under distributional shift

🔗 [MedicalImageAnalysis](https://github.com/KHOUTAIBI/MedicalImageAnalysis)

---

### 🔍 Representation Learning & Computer Vision

> **Self-supervised and contrastive learning for visual representations**

Explores modern representation learning approaches that do not require full supervision, including self-supervised learning (SSL), knowledge distillation, and attention-based models.

**Key topics:**
- Contrastive learning (SimCLR, MoCo style)
- Self-distillation with no labels (DINO-style ViT)
- Linear probing and few-shot transfer evaluation
- Patch-based representations and attention maps

🔗 [Representation-Learning-Computer-Vision](https://github.com/KHOUTAIBI/Representation-Learning-Computer-Vision)

---

### 🤖 DiffusionPolicy

> **Diffusion models as action policies for robot control**

Applies the conditional score-matching framework to **imitation learning**: instead of predicting a single action, the policy models the full distribution over actions given observations, and samples from it via a reverse diffusion process.

The policy is trained to model $p(a | o)$ where $a$ is a robot action sequence and $o$ is the observation. At inference, actions are sampled by running the reverse diffusion chain conditioned on $o$:

$$a_0 \sim p_\theta(a_0 | o) = \int p(a_T) \prod_{t=1}^{T} p_\theta(a_{t-1} | a_t, o) \, da_{1:T}$$

**Papers:** [Diffusion Policy (Chi et al., RSS 2023)](https://arxiv.org/abs/2303.04137)

🔗 [DiffusionPolicy](https://github.com/KHOUTAIBI/DiffusionPolicy)

---

### 🎮 AI Games

> **AI for Games: planning, search, and reinforcement learning agents**

Covers classical and modern techniques for game-playing AI, including tree-search algorithms and RL-based agents trained via self-play.

**Key topics:**
- Monte Carlo Tree Search (MCTS) and UCB selection:

$$\text{UCB}(s, a) = Q(s, a) + c \sqrt{\frac{\ln N(s)}{N(s, a)}}$$

- Policy and value network integration (AlphaZero style)
- Minimax with alpha-beta pruning
- RL agents trained via self-play or reward shaping

🔗 [AI-games](https://github.com/KHOUTAIBI/AI-games)

---

### ⚡ LLM KV-Cache & LoRA Efficiency

> **Memory and compute efficiency for large language model inference and fine-tuning**

Focuses on making LLMs practical at scale, exploring two orthogonal axes: efficient **inference** via KV-cache compression, and efficient **fine-tuning** via low-rank adaptation.

**KV-cache compression** reduces the memory footprint of autoregressive decoding. The key-value states at each layer grow with sequence length $L$ and accumulate across heads $H$:

$$\text{KV cache size} \propto 2 \times H \times L \times d_{\text{head}} \times \text{numlayers}$$

Pruning or quantising these caches is essential for long-context inference.

**LoRA** (Low-Rank Adaptation) freezes the pre-trained weights $W_0$ and adds a trainable low-rank decomposition:

$$W = W_0 + BA, \quad B \in \mathbb{R}^{d \times r}, \quad A \in \mathbb{R}^{r \times k}, \quad r \ll \min(d, k)$$

This reduces the number of trainable parameters from $d \times k$ to $r(d + k)$.

**Papers:** [LoRA (Hu et al., ICLR 2022)](https://arxiv.org/abs/2106.09685)

🔗 [LLM-kv-cache-LoRA-efficiency](https://github.com/KHOUTAIBI/LLM-kv-cache-LoRA-efficiency)

---

### 🧠 Controlled LLM Generation

> **Constrained and guided decoding for controllable text generation**

Explores methods that enforce hard or soft constraints on LLM outputs at generation time, without retraining the model. Key approaches include:

- **Classifier-free guidance** adapted to language models
- **Constrained beam search** with logit manipulation
- **Energy-based decoding** where $p(y | x, c) \propto p_{\text{LM}}(y | x) \cdot \exp(f_c(y))$
- **PPLM / FUDGE-style** plug-in classifiers that steer generation token-by-token

The core idea is to express the desired output distribution as:

$$p_{\text{controlled}}(y) \propto p_{\text{LM}}(y) \cdot \exp(\lambda \cdot s(y, c))$$

where $s(y, c)$ scores how well $y$ satisfies the constraint $c$.

🔗 [controlled-LLM-generation](https://github.com/KHOUTAIBI/controlled-LLM-generation)

---

### 💻 GPU LLM Flash Attention

> **CUDA kernel engineering and FlashAttention for efficient transformer inference**

Hands-on GPU programming applied to the bottleneck of transformer inference: the attention mechanism. Vanilla attention is $O(N^2)$ in memory with respect to sequence length $N$:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

**FlashAttention** rewrites this computation with tiling to avoid materialising the full $N \times N$ attention matrix, reducing memory complexity to $O(N)$ while maintaining numerical equivalence.

Topics covered:
- Writing and profiling custom CUDA kernels
- Tiled matrix multiplication
- Online softmax and numerically stable reductions
- Benchmarking memory bandwidth and FLOPs

**Papers:** [FlashAttention (Dao et al., NeurIPS 2022)](https://arxiv.org/abs/2205.14135) · [FlashAttention-2 (Dao, ICLR 2024)](https://arxiv.org/abs/2307.08691)

🔗 [gpu_llm_flash-attention](https://github.com/KHOUTAIBI/gpu_llm_flash-attention)

---

### 📈 Timeseries & Graph Learning

> **Structured data learning: temporal sequences and graph-structured inputs**

Covers two complementary domains where standard i.i.d. deep learning assumptions break down.

**Time-series:**
- Autoregressive and sequence-to-sequence models
- State space models (S4, Mamba-style)
- Anomaly detection and forecasting

**Graph learning:**
- Graph Neural Networks (GNN): message passing as $h_v^{(l+1)} = \phi(h_v^{(l)}, \text{AGG}(\{h_u^{(l)} : u \in \mathcal{N}(v)\}))$
- Graph Attention Networks (GAT) with learned edge weights
- Spectral graph convolutions and the graph Laplacian $L = D - A$
- Spatio-temporal graphs that combine both axes

🔗 [timeseries-graph-learning](https://github.com/KHOUTAIBI/timeseries-graph-learning)

---

## 3. Getting Started

### Clone all submodules at once

```bash
git clone --recurse-submodules https://github.com/KHOUTAIBI/MVA-2026.git
cd MVA-2026
```

### Or initialise submodules after cloning

```bash
git clone https://github.com/KHOUTAIBI/MVA-2026.git
cd MVA-2026
git submodule update --init --recursive
```

### Navigate to a specific project

```bash
cd GenerativeImageModeling/
# each submodule has its own README, dependencies, and setup instructions
```

Each submodule is self-contained. Refer to the individual `README.md` inside each folder for setup, dependencies, and usage instructions.

---

## 4. Topics at a Glance

| Submodule | Domain | Key techniques |
|---|---|---|
| `GenerativeImageModeling` | Generative models / Inverse problems | DDPM, DDIM, DPS, PiGDM |
| `HyperspectralImagingUnmixing` | Signal processing / Optimization | NMF, PnP, GS-PnP, simplex constraints |
| `MedicalImageAnalysis` | Medical imaging | Segmentation, classification, UNet |
| `Representation-Learning-Computer-Vision` | Self-supervised learning | SSL, DINO, contrastive learning, ViT |
| `DiffusionPolicy` | Robot learning / Control | Conditional diffusion, imitation learning |
| `AI-games` | Planning / RL | MCTS, UCB, AlphaZero, minimax |
| `LLM-kv-cache-LoRA-efficiency` | LLM efficiency | KV-cache, LoRA, PEFT |
| `controlled-LLM-generation` | Controlled generation | Guided decoding, energy-based LM |
| `gpu_llm_flash-attention` | GPU computing | CUDA kernels, FlashAttention, tiling |
| `timeseries-graph-learning` | Structured data | GNN, GAT, SSM, forecasting |

---

> **Program:** [MVA — Mathématiques, Vision, Apprentissage](https://www.master-mva.com/), ENS Paris-Saclay, 2025–2026
> **Author:** [KHOUTAIBI Iliass](https://github.com/KHOUTAIBI)
