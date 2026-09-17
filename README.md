# Soft Temporal Alignment Distillation

Official research code for **Soft Temporal Alignment Distillation**, a cross-modal knowledge distillation framework for transferring motion knowledge from camera-derived 2D human keypoints to an mmWave radar-based human activity recognition model.

The teacher network uses camera-derived 2D keypoint sequences during training, while the student takes only mmWave point clouds. The teacher and distillation modules are removed after training, so inference is performed by the radar student alone.

## Repository Structure

```text
softKD/
├── KD/          # Soft-Align KD, constrained variant, and KD baselines
├── Student/     # PointNet-BiLSTM radar student models
├── Teacher/     # ResNet18-BiLSTM keypoint teacher models
├── mmfi.py      # MM-Fi dataset loader and batch collation
└── config.yaml  # Five-fold cross-subject data split
```

Important training scripts:

- `KD/KD_Align.py`: main Soft-Align KD method.
- `KD/KD_Align_Constrained_warmup.py`: order-and-diversity constrained variant.
- `KD/KD_Align_LSKD.py` and `KD/KD_Align_WKDL.py`: recent logit-level KD baselines.
- `KD/KD_Feature.py`, `KD/KD_AT.py`, and `KD/KD_Feature_AT.py`: feature and attention-transfer baselines.

## Requirements

- Python 3.9 or later
- PyTorch
- torchvision
- NumPy
- SciPy
- OpenCV-Python
- PyYAML
- scikit-learn
- tqdm

Install the main dependencies with:

```bash
pip install torch torchvision numpy scipy opencv-python pyyaml scikit-learn tqdm
```

## Dataset

Download the [MM-Fi dataset](https://github.com/ybhbingo/MMFi_dataset) separately. The dataset is not distributed with this repository.

The code expects the original MM-Fi directory hierarchy:

```text
DATA_ROOT/
└── E01/
    └── S01/
        └── A01/
            ├── rgb/
            ├── mmwave/
            └── ground_truth.npy
```

The default experiment uses 32 development subjects and eight fixed test subjects. The split is defined in `config.yaml`.

## Running Soft-Align KD

Teacher checkpoints are expected in the following form:

```text
TEACHER_ROOT/
├── fold_0/best_res18_pretrain.pth
├── fold_1/best_res18_pretrain.pth
├── fold_2/best_res18_pretrain.pth
├── fold_3/best_res18_pretrain.pth
└── fold_4/best_res18_pretrain.pth
```

The public code uses the following repository-relative paths by default:

- configuration: `config.yaml`
- MM-Fi dataset: `data/MMFi`
- teacher checkpoints: `checkpoints/teacher`
- experiment outputs: `outputs/soft_align`

You may either arrange the files using these paths or edit the corresponding
arguments at the bottom of `KD/KD_Align.py`.

Then run from the repository root:

```bash
export PYTHONPATH="$PWD:$PWD/Student:$PWD/Teacher"
python KD/KD_Align.py
```

The script trains all five development folds, saves the best student checkpoint for each fold, and writes the aggregate accuracy and macro-F1 results to the output directory.

The other KD scripts can be run in the same way after updating their dataset, checkpoint, and output paths.

## Notes

- Camera-derived 2D keypoints are used only during teacher training and knowledge distillation.
- Deployment uses only the mmWave radar student.
- Model checkpoints and datasets are not included in this repository.
- The code is released for academic research and reproducibility. A formal license should be added before redistribution.

## Citation

Citation information will be added after publication.
