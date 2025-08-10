# Stochastic-Predictive-Maintenance-for-Flexible-Manufacturing-Systems
This shows how to preprocess, augment, and analyze CNC milling data for stochastic predictive maintenance. The workflow includes data selection, extrapolation, and final analysis using provided Python scripts.
Steps to Reproduce the Results
1. Data Preprocessing and Augmentation
Source Data:
The original dataset (mill_data.csv) contains 16 iterations of CNC milling sensor readings.
Random Selection:
We randomly selected 4 iterations from the original data and saved them as mill_data2.csv.
Augmentation/Extrapolation:
Data augmentation and extrapolation were performed using the augmentation code provided in this repository. This step prepares the data for robust analysis and simulates longer operational periods.
2. Data Selection for Analysis
Final Dataset:
Out of the 4 iterations in mill_data2.csv, we chose 3 iterations for the final analysis. These are saved in extrapolated_data.csv.
3. Running the Analysis
Use the main analysis script to process extrapolated_data.csv.
The script will:
Fit Weibull models,
Calculate Health Indicators (HI) and Remaining Useful Life (RUL),
Trigger maintenance events,
Output results and visualizations for each iteration.

Files
mill_data.csv – Full original dataset (16 iterations)
mill_data2.csv – Randomly selected 4 iterations 
extrapolated_data.csv – Final 3 iterations used for analysis
augmentation.py – Script for data preprocessing and extrapolation 
analysis.py – Main analysis and visualization script
Repository structure-
├── Data Augmentation Code
│ 
│ ├── mill_data2.csv # Selected 4 iterations (pre-augmentation)
│ └── augmentation.py # Data preprocessing & extrapolation script
└── NASA milling dataset
│  ├── mill_data.csv # Raw dataset (16 iterations)
├── SPM Modelling Code
│ ├── extrapolated_data.csv # Final 3 iterations for analysis
│ └── analysis.py # Main analysis & visualization script
├── Results/ # Auto-generated outputs
│ ├── Weibull_plots/ # PDF/CDF visualizations
│ ├── HI_RUL_trends/ # Dual-axis monitoring charts
│ 
└── README.md # This documentation

Requirements
Python 3.x
pandas
numpy
scipy
matplotlib
