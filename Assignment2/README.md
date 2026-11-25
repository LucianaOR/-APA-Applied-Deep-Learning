# Applied Deep Learning — Assignment 2

This repository contains the implementation of **Assignment 2** for the Applied Deep Learning course (APA). The assignment explores deep learning techniques for image representation, object detection, and object re-identification.

## Project Overview

The assignment is divided into two main parts.

### Part I — Autoencoders

This part investigates different autoencoder architectures using the **Kuzushiji-MNIST (KMNIST)** dataset.

The implemented models are:

- **Autoencoder (AE):** learns a compressed representation of the input data and reconstructs the original images.
- **Variational Autoencoder (VAE):** learns a probabilistic latent representation.
- **Denoising Autoencoder (DAE):** learns to reconstruct clean images from noisy inputs.

The experiments include:

- Model training and reconstruction loss analysis.
- Comparison of reconstructed images.
- Latent space visualization using t-SNE.
- Evaluation of learned representations through a classification task.

### Part II — Object Detection and Unique Object Counting

This part focuses on detecting and identifying people in images and video sequences using the **MID-3K RGB** dataset.

#### Part II-A — YOLO Object Detection

Two YOLO architectures are evaluated:

- YOLOv8s
- YOLOv10s

Both architectures are trained under two initialization strategies:

- **From scratch:** models initialized from architecture configuration files.
- **Pretrained:** models initialized using pretrained weights.

Experiments are conducted using two dataset splits:

- **Random:** stratified random split, with training and testing samples from the same departments.
- **DEEC_DEI:** training on DEEC data and testing on DEI data, evaluating generalization across departments.

The evaluation considers:

- Precision
- Recall
- mAP@50
- mAP@50–95
- Training and validation loss curves

#### Part II-B — Unique Object Counting

This part addresses the identification and counting of unique people in a video sequence.

The objective is to combine object detection, tracking, and appearance-based re-identification to maintain consistent identities across frames, including situations involving temporary occlusions or people leaving and re-entering the scene.

The final analysis includes the number of unique individuals identified and their identity associations throughout the video.

## Repository Structure

```text
Assignment2/
│
├── PA2.ipynb
├── README.md
├── Trabalho 2 APA 2026.pdf
│
├── data/
│   ├── mid3k_random.yaml
│   └── mid3k_deec_dei.yaml
│
├── APA2026_MID-3K/
│   ├── Random/
│   └── DEEC_DEI/
│
├── MID-3K-rgb/
│   ├── images/
│   └── labels/
│
├── figures/
│   ├── part1/
│   └── part2/
│
├── results/
│
└── runs/
```

*Note: Generated files, model checkpoints, and large dataset files may be excluded from version control.*

## Dataset

### Kuzushiji-MNIST

The KMNIST dataset is used in Part I to train and evaluate the autoencoder models.

It consists of 28 × 28 grayscale images representing 10 classes of Japanese cursive characters.

### MID-3K RGB

The MID-3K dataset is used in Part II for person detection.

It contains RGB images with YOLO-format bounding-box annotations.

Each annotation follows the format:

```text
class_id x_center y_center width height
```

The coordinates are normalized relative to the image dimensions. Class `0` represents a person.

The dataset and split manifests must be available locally before running the corresponding experiments. The image dataset is not included in this repository because of its size.

## Requirements

The experiments were developed using Python and PyTorch, with support for Apple Silicon acceleration through Metal Performance Shaders (MPS).

Main dependencies include:

- Python
- PyTorch
- Torchvision
- Ultralytics
- NumPy
- Pandas
- Matplotlib
- OpenCV
- Scikit-learn
- Pillow

Install the required packages using:

```bash
pip install torch torchvision ultralytics numpy pandas matplotlib opencv-python scikit-learn pillow
```

Additional dependencies may be required depending on the notebook cells and the execution environment.

## Configuration and Execution

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd <repository-name>/Assignment2
   ```

2. Install the required dependencies.

3. Ensure that the datasets are available in the expected local directories.

4. Check the dataset configuration files in `data/` and adjust their paths to match the local environment.

5. Open `PA2.ipynb` in Jupyter Notebook or JupyterLab.

6. Execute the notebook cells in order, following the section structure and experimental workflow.

The notebook contains the implementation, training procedures, evaluation, and visualizations for both parts of the assignment.

## Experimental Outputs

Training checkpoints and experiment logs are generated under `runs/`.

Evaluation results, figures, and other relevant outputs are organized under `results/` and `figures/`.

These directories contain generated artifacts and are not required to be committed to the repository in their entirety.

## Reproducibility

The experiments use fixed random seeds where applicable to improve reproducibility.

For the YOLO experiments, the main training configuration includes:

- Epochs: 25
- Image size: 640 × 640
- Batch size: 8
- Random seed: 42
- Device: Apple Metal Performance Shaders (MPS), when available

Results may vary depending on the hardware, software versions, and execution environment.

## Authors

**Applied Deep Learning — Assignment 2**

Academic project developed as part of the Applied Deep Learning course.