# Overview

This project explores how machine learning can make eye-gaze estimation more reliable across different individuals while reducing the need for extensive user calibration.

Using the MPIIGaze dataset, the project investigates personalized gaze prediction, generalization to unseen participants, and the effect of limited calibration data. It also examines spatial variations in prediction errors and their implications for gaze-based interface design.

# Objectives

Evaluate gaze prediction under personalized and subject-independent settings.

Compare different regression approaches against baseline methods.

Investigate low-shot calibration using 1, 3, 5, and 9 calibration points.

Analyze prediction errors across different screen regions.

Study implications for virtual keyboard button sizing using a Fitts' Law-based audit.

## Repository Structure

```text
es678_gaze_tracking/
├── data/                       # MPIIGaze annotation files (p00..p14) & screenSize.mat
├── results/                    # Generated CSV evaluation tables & high-DPI PNG figures
│   ├── within_subject_results.csv
│   ├── loso_calibration_results.csv
│   ├── fitts_law_ui_audit.csv
│   ├── fig1_calibration_curve.png
│   └── fig2_spatial_heatmap.png
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # MPIIGaze parser, pixel-to-mm screen calibration & fallback
│   ├── features.py             # 20D roll-aligned, scale-normalized binocular + head-pose features
│   ├── models.py               # Baselines, Ridge, Kernel Ridge, SVR & Empirical Bayes Affine Calibrator
│   ├── evaluate.py             # 5-fold within-subject CV, 15-subject LOSO-CV & Fitts's Law audit
│   └── visualize.py            # Generates calibration curves and spatial error heatmaps
├── main.py                     # Single-command reproduction entry point
└── README.md
```
# Dataset

The project uses the MPIIGaze dataset, containing approximately 213,659 images from 15 participants.

Dataset: MPIIGaze on Kaggle

URL: https://www.kaggle.com/datasets/dhruv413/mpiigaze

# Methodology

The project follows three main stages:

Feature Extraction: Extracts raw pupil coordinates and geometric eye-related information.

Model Evaluation: Compares baseline and machine-learning approaches using five-fold within-subject cross-validation and 15-subject Leave-One-Subject-Out (LOSO) validation.

Calibration and Spatial Analysis: Evaluates low-shot calibration and investigates how prediction errors vary across screen regions.

# Key Results

Personalized prediction: Kernel Ridge achieves the lowest mean error of 54.66 mm.

Unseen participants: Ridge Regression achieves the lowest LOSO error of 67.02 mm.

Low-shot calibration: Ridge error decreases to 64.37 mm using nine calibration points.

Spatial analysis: Prediction error is higher near screen boundaries than at the centre.

# Running the Project
## Environment Setup

This project is built entirely in Python (3.10+) and runs efficiently on a standard CPU (no GPU required).
### 1. Clone the Repository
```bash
git clone [https://github.com/niharpasi/es678-gaze-tracking.git](https://github.com/niharpasi/es678-gaze-tracking.git)
cd es678-gaze-tracking
```

### 2. Install Dependencies
```
python -m pip install --upgrade pip
python -m pip install numpy>=1.24.0 pandas>=2.0.0 scipy>=1.10.0 scikit-learn>=1.3.0 matplotlib>=3.7.0 jinja2>=3.1.2
```

### 3. Reproducibility & Data Access
```
Download the dataset
```
Because our pipeline operates directly on pre-extracted facial landmarks, 3D head pose vectors, and physical screen coordinates, you do not need to store the 2.2 GB of raw .jpg eye images. Place the lightweight annotation text files inside ./data/ following this layout:
```
es678_gaze_tracking/
└── data/
    ├── Annotation Subset/          # Manual 16-point facial & pupil coordinates (p00.txt .. p14.txt)
    └── Data/
        └── Original/
            ├── p00/
            │   └── day01/annotation.txt   # Ground-truth screen (x, y) & 3D head pose
            ├── ...
            └── p14/
```
for this Open Command Prompt (cmd) inside your downloaded MPIIGaze folder on your laptop and run these two commands to copy only the .txt files (ignoring all .jpg images):
```
robocopy "Annotation Subset" "mpiigaze_complete_text\Annotation Subset" *.txt /S
robocopy "Data\Original" "mpiigaze_complete_text\Data\Original" annotation.txt /S
```

### 4. Reproducing All Results & Figures
```
python main.py
```

# AI-use disclosure
The MPIIGaze dataset (download link has been provided above). I used ChatGPT to assist with explanations and report writing, Gemini for code assistance. I independently checked the implementation, evaluation outputs, numerical results, and reported conclusions against the code and experiment outputs.

# References
1) dhruv413, ``MPIIGaze,'' Kaggle. Available: \url{https://www.kaggle.com/datasets/dhruv413/mpiigaze}.

2) D. Cazzato, M. Leo, C. Distante, and H. Voos, ``When I Look into Your Eyes: A Survey on Computer Vision Contributions for Human Gaze Estimation and Tracking,''Sensors, vol. 20, no. 13, p. 3739, 2020. https://doi.org/10.3390/s20133739

3) Ramanand and Y. K. Meena, ``Towards Enhancing Visual Reasoning through Gaze-Semantic Alignment and Attention Correction,'' in \textit{Proceedings of the 2026 Symposium on Eye Tracking Research and Applications (ETRA '26)}, 2026, Art. no. 54, pp. 1--3. https://dl.acm.org/doi/full/10.1145/3797246.3805853

4) A. Bulling et al., ``It's Written All Over Your Face: Full-Face Appearance-Based Gaze Estimation,'' in \textit{Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops}, 2017. https://arxiv.org/pdf/1611.08860

