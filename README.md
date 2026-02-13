# Abnormal Reaction Detection System (Dual Verification: Autoencoder + Isolation Forest)

This repository contains the code for identifying abnormal reactions in photometric absorbance time-series data. The system employs a **Dual Verification** approach:

1. **Phase 1 (Autoencoder)**: A 1D-CNN Autoencoder trained on reaction curves, using MAD-based anomaly thresholding and Cosine Similarity filtering to detect structurally unique patterns.
2. **Phase 2 (Isolation Forest)**: Reaction curves are rendered as images, features are extracted via a pre-trained ResNet50 (ImageNet), and an Isolation Forest detects visual outliers.

Only anomalies detected by **BOTH** methods are flagged as **"True Anomalies"** (Dual Verified), significantly reducing false positives.

## Directory Structure

```
├── Code/                   # Source code modules
│   ├── src/                # Core logic
│   │   ├── config.py           # Output directory management
│   │   ├── data_loader.py      # Data loading & feature extraction
│   │   ├── model_autoencoder.py # Autoencoder architecture
│   │   ├── anomaly_detection.py # MSE thresholding & similarity filter
│   │   ├── visualization.py     # Basic plots
│   │   ├── visualization_enhanced.py # Publication-quality plots
│   │   └── logger.py           # Logging utilities
│   └── requirements.txt
├── Scripts/                # Execution scripts
│   ├── integrated_analysis.py   # Main entry point
│   ├── run_autoencoder.py       # Phase 1: Autoencoder analysis
│   ├── generate_images.py       # Phase 2 prep: Curve → image
│   ├── extract_features.py      # Phase 2 prep: ResNet50 features
│   ├── detect_and_compare.py    # Phase 2: Isolation Forest & merge
│   ├── generate_report.py       # Publication-quality report generator
│   └── verify_alignment.py      # Data alignment verification
├── Data/                   # Input CSV files (not tracked)
└── Results/                # Timestamped output (not tracked)
    └── YYYYMMDD_HHMMSS/
        ├── Data/           # Intermediate results
        ├── Models/         # Saved models & parameters
        ├── Figures/        # Visualizations
        ├── Reports/        # Summary reports & statistics
        └── Logs/           # Execution logs
```

## Setup

1. Python >= 3.10 with TensorFlow >= 2.10 is required.
2. Install dependencies:
   ```bash
   pip install -r Code/requirements.txt
   ```

## Usage

1. Place your data CSV file in `Data/`. The CSV should contain columns `RAM11`–`RAM38` (Main absorbance) and `RAS11`–`RAS38` (Sub absorbance).
2. Update the `DEFAULT_CSV_PATH` in `Scripts/integrated_analysis.py` to point to your CSV file.
3. Run the integrated analysis:
   ```bash
   python Scripts/integrated_analysis.py
   ```
4. Results are saved in `Results/<timestamp>/Reports/<dataset>/Summary_Report.md`.

## Analysis Modes

| Mode | Data | Description |
| :--- | :--- | :--- |
| Main | RAM (Main Absorbance) | Primary reaction measurement |
| Sub  | RAS (Sub Absorbance)  | Secondary wavelength measurement |
| Calc | RAM − RAS (Difference)| Calculated difference signal |

## Configuration

Key parameters can be adjusted in `Scripts/run_autoencoder.py`:

| Parameter | Default | Description |
| :--- | :---: | :--- |
| `EPOCHS` | 50 | Training epochs |
| `BATCH_SIZE` | 32 | Training batch size |
| `SIMILARITY_THRESHOLD` | 0.92 | Cosine similarity for anomaly filtering |
| `MIN_NEIGHBORS` | 3 | Minimum neighbors to classify as cluster |

Isolation Forest parameters are in `Scripts/detect_and_compare.py`:

| Parameter | Default | Description |
| :--- | :---: | :--- |
| `n_estimators` | 100 | Number of trees |
| `contamination` | 0.02 | Expected anomaly ratio |

## License

This work is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

[![CC BY-NC-SA 4.0](https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
