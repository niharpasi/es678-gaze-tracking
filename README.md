# Overview

This project explores how machine learning can make eye-gaze estimation more reliable across different individuals while reducing the need for extensive user calibration.

Using the MPIIGaze dataset, the project investigates personalized gaze prediction, generalization to unseen participants, and the effect of limited calibration data. It also examines spatial variations in prediction errors and their implications for gaze-based interface design.

# Objectives

Evaluate gaze prediction under personalized and subject-independent settings.

Compare different regression approaches against baseline methods.

Investigate low-shot calibration using 1, 3, 5, and 9 calibration points.

Analyze prediction errors across different screen regions.

Study implications for virtual keyboard button sizing using a Fitts' Law-based audit.

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
Clone the repository.

Run the main script from the project directory:

python main.py

# AI-use disclosure
The MPIIGaze dataset (download link has been provided above). I used ChatGPT to assist with explanations and report writing, Gemini for code assistance. I independently checked the implementation, evaluation outputs, numerical results, and reported conclusions against the code and experiment outputs.


