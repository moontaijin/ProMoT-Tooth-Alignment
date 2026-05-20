# ProMoT: Tooth Alignment via Virtual Trajectories with Clinical Constraints

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

Official PyTorch implementation of:

> **Tooth Alignment via Virtual Trajectories with Clinical Constraints**

**ProMoT** models automatic tooth alignment as a **virtual progressive tooth movement trajectory** rather than a one-shot final-pose regression problem.  
Given pre-treatment tooth geometries, the model predicts staged incremental rigid transformations and evaluates clinical constraints at intermediate states.

<p align="center">
  <img src="assets/method_overview.png" width="900">
</p>

---

## News

- Code and pretrained checkpoints will be released soon.
- Dataset preparation instructions are provided under [`data/README.md`](data/README.md).

---

## Highlights

- **Process-centric tooth alignment** through virtual staged movement prediction
- **K-step incremental 6-DoF rigid transformation modeling** in SE(3)
- **SSM-based sequential predictor** for progressive refinement
- **Trajectory Stabilization Strategy (TSS)** for stable intermediate trajectories
- **Intermediate-state self-learning** for denser supervision from partially progressed states
- **Stage-wise clinical constraints** for collision, occlusal-plane, and spacing evaluation
- End-to-end training from **pre-/post-treatment IOS data only**

---

## Overview

Most automatic tooth alignment methods directly regress a final rigid transformation for each tooth. This one-shot formulation can be insufficient for complex tooth movements, because orthodontic treatment is typically performed through repeated staged refinements.

ProMoT reformulates tooth alignment as a progressive trajectory prediction problem. Starting from the initial tooth arrangement, the model predicts a sequence of incremental transformations:

```text
Initial state -> Stage 1 -> Stage 2 -> ... -> Stage K -> Final aligned state
```

At each stage, the intermediate arrangement can be evaluated using clinically motivated constraints. The final prediction is obtained by cumulatively composing all incremental transformations.

<p align="center">
  <img src="assets/trajectory_visualization.png" width="900">
</p>

---

## Method

The full framework consists of four main components.

### 1. Progressive Tooth Movement Modeling

For each tooth, ProMoT predicts a sequence of incremental rigid motions.  
Each stage produces an SE(3) update, and the final transformation is obtained by cumulative composition.

<p align="center">
  <img src="assets/progressive_modeling.png" width="850">
</p>

### 2. SSM-based Sequential Prediction

A PointNet++ encoder extracts stage-wise tooth features from the transformed tooth point cloud.  
An SSM-based sequential model then predicts the next incremental tooth movement, allowing information to be propagated across refinement stages.

### 3. Trajectory Stabilization Strategy

Training a staged trajectory from only pre-/post-treatment pairs can be unstable.  
ProMoT uses trajectory stabilization to encourage smooth, bounded, and continuous intermediate movements during early training.

### 4. Intermediate-state Self-learning

Generated intermediate states are reused as new starting states.  
The model is then trained to predict the remaining motion from those partially progressed configurations, which densifies supervision along the virtual trajectory.

### 5. Stage-wise Clinical Constraints

Clinical constraints are evaluated at every intermediate stage, including:

- inter-tooth collision and interpenetration,
- occlusal-plane consistency,
- excessive interproximal spacing.

<p align="center">
  <img src="assets/framework.png" width="900">
</p>

---

## Repository Structure

```text
ProMoT-Tooth-Alignment/
│
├── README.md
├── LICENSE
├── requirements.txt
├── environment.yml
│
├── assets/
│   ├── method_overview.png
│   ├── framework.png
│   ├── progressive_modeling.png
│   ├── trajectory_visualization.png
│   └── qualitative_results.png
│
├── configs/
│   ├── default.yaml
│   ├── train.yaml
│   ├── train_upper.yaml
│   ├── train_lower.yaml
│   ├── eval.yaml
│   ├── eval_upper.yaml
│   └── eval_lower.yaml
│
├── data/
│   ├── README.md
│   └── splits/
│       ├── fold_0.json
│       ├── fold_1.json
│       ├── fold_2.json
│       ├── fold_3.json
│       └── fold_4.json
│
├── src/
│   ├── datasets/
│   │   ├── dataset.py
│   │   ├── preprocessing.py
│   │   └── transforms.py
│   │
│   ├── models/
│   │   ├── pointnet2_encoder.py
│   │   ├── ssm_predictor.py
│   │   ├── promot.py
│   │   └── losses.py
│   │
│   ├── constraints/
│   │   ├── collision.py
│   │   ├── occlusal_plane.py
│   │   └── spacing.py
│   │
│   └── utils/
│       ├── geometry.py
│       ├── metrics.py
│       ├── se3.py
│       ├── visualization.py
│       └── io.py
│
├── scripts/
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   ├── inference.py
│   ├── visualize_trajectory.py
│   └── export_results.py
│
├── notebooks/
│   ├── demo_inference.ipynb
│   ├── visualize_trajectory.ipynb
│   └── encoder_feature_clustering.ipynb
│
├── checkpoints/
│   └── README.md
│
└── outputs/
```

---

## Installation

### 1. Clone this repository

```bash
git clone https://github.com/your-username/ProMoT-Tooth-Alignment.git
cd ProMoT-Tooth-Alignment
```

### 2. Create the environment

Using `conda`:

```bash
conda create -n promot python=3.10
conda activate promot
pip install -r requirements.txt
```

Or using the provided environment file:

```bash
conda env create -f environment.yml
conda activate promot
```

---

## Dataset Preparation

### Expected Dataset Structure

The expected raw dataset structure is:

```text
dataset_root/
├── data_001/
│   ├── crown/
│   │   ├── up/
│   │   │   ├── 1.stl
│   │   │   ├── 2.stl
│   │   │   └── ...
│   │   └── down/
│   │       ├── 17.stl
│   │       ├── 18.stl
│   │       └── ...
│   │
│   ├── tooth/
│   │   ├── up/
│   │   └── down/
│   │
│   └── target_matrix.txt
│
├── data_002/
├── data_003/
└── ...
```

Each case should contain:

- tooth-wise STL meshes,
- upper and lower arch folders,
- target transformation matrices,
- case-level metadata if available.

Detailed instructions are provided in [`data/README.md`](data/README.md).

> Raw clinical data are not included in this repository. Please prepare the data following the required structure.

---

## Preprocessing

To convert STL meshes into sampled point clouds:

```bash
python scripts/preprocess.py \
    --data_root /path/to/dataset_root \
    --output_root /path/to/processed_data \
    --num_points 512
```

The processed dataset will be saved as:

```text
processed_data/
├── data_001/
│   ├── points.npy
│   ├── tooth_ids.npy
│   ├── centers.npy
│   ├── target_matrix.npy
│   └── meta.json
├── data_002/
└── ...
```

---

## Training

### Train ProMoT

```bash
python scripts/train.py \
    --config configs/train.yaml \
    --data_root /path/to/processed_data \
    --output_dir outputs/promot
```

### Train upper or lower arch separately

```bash
python scripts/train.py \
    --config configs/train_upper.yaml \
    --data_root /path/to/processed_data \
    --output_dir outputs/promot_upper
```

```bash
python scripts/train.py \
    --config configs/train_lower.yaml \
    --data_root /path/to/processed_data \
    --output_dir outputs/promot_lower
```

### Example Configuration

```yaml
seed: 42

data:
  data_root: /path/to/processed_data
  split_file: data/splits/fold_0.json
  arch: full
  num_teeth: 32
  num_points: 512

model:
  name: promot
  encoder: pointnet2
  sequential_model: ssm
  hidden_dim: 256
  num_stages: 21

training:
  epochs: 500
  batch_size: 32
  learning_rate: 1.5e-4
  weight_decay: 1.0e-4
  optimizer: adam

loss:
  lambda_add: 1.0
  lambda_aae: 1.0
  lambda_tss: 1.0
  lambda_clinical: 1.0
  lambda_inter: 1.0
  lambda_occ: 1.0
  lambda_spacing: 1.0

self_learning:
  enabled: true
  start_epoch: 200
  micro_batch_size: 3

output:
  save_dir: outputs/promot
  save_interval: 50
```

---

## Evaluation

To evaluate a trained checkpoint:

```bash
python scripts/evaluate.py \
    --config configs/eval.yaml \
    --checkpoint checkpoints/promot.pth \
    --data_root /path/to/processed_data
```

The evaluation script reports:

- ADD/AUC,
- ADD,
- AAE,
- mean rotation error,
- mean translation error,
- optional clinical constraint measurements.

Example output:

```text
ADD/AUC      : 0.8631
ADD          : 0.7938 mm
AAE          : 0.9802 deg
ME_Rot       : 3.1209 deg
ME_Trans     : 1.7502 mm
```

---

## Inference

Run inference on a single case:

```bash
python scripts/inference.py \
    --case_dir /path/to/processed_data/data_001 \
    --checkpoint checkpoints/promot.pth \
    --output_dir outputs/inference/data_001
```

The inference output contains:

```text
outputs/inference/data_001/
├── pred_final_transform.npy
├── pred_trajectory.npy
├── aligned_teeth/
├── intermediate_states/
└── metrics.json
```

---

## Visualization

Visualize the predicted virtual tooth movement trajectory:

```bash
python scripts/visualize_trajectory.py \
    --pred_dir outputs/inference/data_001 \
    --output_dir outputs/visualization/data_001
```

<p align="center">
  <img src="assets/qualitative_results.png" width="900">
</p>

---

## Pretrained Checkpoints

Pretrained checkpoints will be released separately.

| Model | Description | Checkpoint |
|---|---|---|
| ProMoT | Full dentition model | Coming soon |
| ProMoT-Upper | Upper arch model | Coming soon |
| ProMoT-Lower | Lower arch model | Coming soon |

After downloading checkpoints, place them under:

```text
checkpoints/
├── promot.pth
├── promot_upper.pth
└── promot_lower.pth
```

---

## Results

### Quantitative Comparison

| Method | ADD/AUC ↑ | ADD ↓ | AAE ↓ | ME<sub>Rot</sub> ↓ | ME<sub>Trans</sub> ↓ |
|---|---:|---:|---:|---:|---:|
| TANet | 0.681 | 1.476 | 1.894 | 8.894 | 3.400 |
| TAlignNet | 0.704 | 1.426 | 1.773 | 7.776 | 2.892 |
| PSTN | 0.754 | 1.231 | 1.221 | 11.922 | 3.586 |
| STTAlign | 0.844 | 0.820 | 1.136 | 3.136 | 1.836 |
| **ProMoT** | **0.863** | **0.794** | **0.980** | **3.121** | **1.750** |

### Ablation Study

| Method | ADD/AUC ↑ | ADD ↓ | AAE ↓ | ME<sub>Rot</sub> ↓ | ME<sub>Trans</sub> ↓ |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.6421 | 1.6206 | 2.1203 | 12.3007 | 3.9505 |
| + Progressive modeling | 0.7650 | 1.0806 | 1.4208 | 5.2001 | 2.1504 |
| + Stage-wise constraints | 0.8107 | 0.9501 | 1.2803 | 4.3002 | 1.9801 |
| + Trajectory module | 0.8267 | 0.9141 | 1.1303 | 3.7662 | 1.9241 |
| + Self-learning | **0.8631** | **0.7938** | **0.9802** | **3.1209** | **1.7502** |

---

## Important Note on Virtual Trajectories

The intermediate states predicted by ProMoT are **virtual progressive tooth movement trajectories** learned from pre-/post-treatment observations. They are not directly supervised by clinically observed intermediate scans.

Therefore, the intermediate trajectory should be interpreted as a model-generated staged refinement that enables process-centric alignment and stage-wise clinical constraint evaluation.

---

## Citation

If you find this repository useful, please cite our paper:

```bibtex
@inproceedings{promot2026toothalignment,
  title     = {Tooth Alignment via Virtual Trajectories with Clinical Constraints},
  author    = {Anonymous Authors},
  booktitle = {Proceedings of the International Conference on Medical Image Computing and Computer-Assisted Intervention},
  year      = {2026}
}
```

The BibTeX entry will be updated after publication.

---

## Acknowledgements

This repository builds upon prior studies on automatic tooth alignment, point-cloud representation learning, and state-space sequence modeling.

---

## License

This project is released under the MIT License. See [`LICENSE`](LICENSE) for details.

---

## Contact

For questions, please open an issue or contact the authors after publication.
