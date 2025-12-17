# F1 Data Analysis Project

A comprehensive machine learning project analyzing Formula 1 race data to predict lap times, optimize pit stop strategy, and cluster driver performance.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Status](https://img.shields.io/badge/Status-Completed-success.svg)]()

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Data](#data)

---

## 🎯 Overview

This project applies machine learning techniques to Formula 1 race data from the 2023 season, addressing three critical problems in motorsports analytics:

1. **Lap Time Prediction** (Regression) - Predict lap times based on tire compound and temperature
2. **Pit Stop Strategy Optimization** (Classification) - Classify optimal pit stop windows
3. **Driver Performance Clustering** (Unsupervised Learning) - Group drivers by performance characteristics

### Key Achievements

- ✅ **92% accuracy** in lap time prediction (R² = 0.92)
- ✅ **100% accuracy** in pit stop prediction
- ✅ **3 meaningful driver clusters** identified
- ✅ **9 machine learning models** evaluated
- ✅ **8,763 laps** analyzed from 8 races

---

## 📁 Project Structure

```
F1-Analysis/
│
├── 📄 README.md                          # This file
├── 📄 requirements.txt                   # Python dependencies
│
├── 📓 F1_Analysis_Project.ipynb          # Complete Jupyter notebook (main file)
│
├── 📊 Datasets (CSV)
│   ├── raw_laps.csv                      # Raw combined lap data (8,763 laps)
│   ├── laptime_dataset.csv               # Preprocessed data for regression
│   ├── pit_dataset.csv                   # Preprocessed data for classification
│   ├── driver_stats.csv                  # Driver performance statistics
│   └── driver_clusters.csv               # Driver clustering results
│
└── 📈 charts/                            # Generated visualizations
    ├── chart1_lap_time_distribution.png  # Lap time histogram
    ├── chart2_top_drivers.png            # Top 10 drivers by average lap time
    ├── chart3_temperature_effects.png    # Temperature vs lap time analysis
    ├── chart4_pit_stop_patterns.png      # Pit stop frequency and patterns
    ├── chart5_driver_clustering.png      # Driver clusters visualization
    ├── chart6_race_comparison.png        # Performance across races
    └── chart7_correlation_heatmap.png    # Feature correlation matrix
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Internet connection (for initial data download)

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- `fastf1` - F1 data API
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `scikit-learn` - Machine learning
- `xgboost` - Gradient boosting
- `matplotlib` - Plotting
- `seaborn` - Statistical visualization
- `scipy` - Scientific computing
- `jupyter` - Interactive notebooks

### Step 2: Verify Installation

```bash
python -c "import fastf1, pandas, sklearn, xgboost; print('✅ All packages installed successfully!')"
```

---

## ▶️ How to Run

### Option 1: Using Jupyter Notebook (Recommended)

```bash
# 1. Start Jupyter Notebook
jupyter notebook

# 2. In your browser, click on F1_Analysis_Project.ipynb

# 3. Run all cells: Menu → Cell → Run All
#    Or use Shift + Enter to run cells one by one
```

### Option 2: Using JupyterLab

```bash
# 1. Install JupyterLab (if not already installed)
pip install jupyterlab

# 2. Start JupyterLab
jupyter lab

# 3. Open F1_Analysis_Project.ipynb from the file browser
# 4. Run all cells: Menu → Run → Run All Cells
```

### Option 3: Using VS Code

```bash
# 1. Open VS Code in the project directory
code .

# 2. Install the Jupyter extension (if not already installed)
# 3. Click on F1_Analysis_Project.ipynb
# 4. Click "Run All" button at the top of the notebook
```

### What Happens When You Run

Running the notebook will:
1. **Load data** from CSV files (or download if first time)
2. **Preprocess data** into 3 datasets (laptime, pit stop, driver stats)
3. **Train 9 machine learning models** (3 regression, 3 classification, 3 clustering)
4. **Evaluate and compare** all models with comprehensive metrics
5. **Display results** directly in the notebook cells

**Note**: The charts are pre-generated and stored in the `charts/` folder. The datasets are also pre-processed and ready to use.

### Keyboard Shortcuts

- **Shift + Enter**: Run current cell and move to next
- **Ctrl + Enter**: Run current cell and stay in current cell
- **Cell → Run All**: Run all cells from beginning

---

## 📊 Results

### Model Performance Summary

#### Lap Time Prediction (Regression)
| Model | RMSE | MAE | R² |
|-------|------|-----|-----|
| Linear Regression | 14.925s | 11.690s | 0.1287 |
| Random Forest | 4.573s | 2.011s | 0.9182 |
| **XGBoost** ⭐ | **4.489s** | **1.995s** | **0.9212** |

#### Pit Stop Prediction (Classification)
| Model | Accuracy | Precision | Recall | F1 | AUC-ROC |
|-------|----------|-----------|--------|-----|---------|
| Logistic Regression | 97.8% | 98.5% | 97.7% | 98.1% | 0.9983 |
| Random Forest | 100% | 100% | 100% | 100% | 1.0000 |
| **Gradient Boosting** ⭐ | **100%** | **100%** | **100%** | **100%** | **1.0000** |

#### Driver Clustering
| Algorithm | Silhouette Score | Davies-Bouldin Index | Calinski-Harabasz Index |
|-----------|------------------|----------------------|-------------------------|
| K-Means | 0.4068 | 0.7781 | 144.56 |
| Hierarchical | 0.4068 | 0.7781 | 144.56 |
| DBSCAN | N/A | N/A | N/A |

### Key Insights

1. **Tire compound and temperature explain 92% of lap time variance**
2. **Pit stop decisions are highly predictable** based on tire age and lap number
3. **Three distinct driver performance groups** identified:
   - Top performers (13 drivers)
   - Mid-tier (7 drivers)
   - Outlier (1 driver)

---

## 🛠️ Technologies Used

### Core Libraries
- **FastF1** (v3.6+) - F1 data API
- **Pandas** (v2.0+) - Data manipulation
- **NumPy** (v1.24+) - Numerical computing
- **Scikit-learn** (v1.3+) - Machine learning
- **XGBoost** (v2.0+) - Gradient boosting

### Visualization
- **Matplotlib** (v3.7+) - Plotting
- **Seaborn** (v0.12+) - Statistical visualization

### Additional
- **SciPy** (v1.10+) - Scientific computing
- **Jupyter** - Interactive notebooks

---

## 📈 Data

### Dataset Overview

- **Total Laps**: 8,763
- **Races Analyzed**: 8 (2023 Formula 1 season)
- **Drivers**: 21
- **Features**: 34 columns per lap

### Races Included

1. Bahrain Grand Prix
2. Saudi Arabian Grand Prix
3. Australian Grand Prix
4. Azerbaijan Grand Prix
5. Spanish Grand Prix
6. Canadian Grand Prix
7. Austrian Grand Prix
8. Belgian Grand Prix

### CSV Files Description

- **`raw_laps.csv`**: Complete raw data from all races (8,763 laps, 34 columns)
- **`laptime_dataset.csv`**: Preprocessed data for lap time regression model
- **`pit_dataset.csv`**: Preprocessed data for pit stop classification model
- **`driver_stats.csv`**: Aggregated driver performance statistics (21 drivers)
- **`driver_clusters.csv`**: Driver clustering results with cluster labels

### Visualizations

All 7 charts are saved in the `charts/` folder as PNG images:
- Chart 1: Lap time distribution histogram
- Chart 2: Top 10 drivers by average lap time
- Chart 3: Temperature effects on lap times
- Chart 4: Pit stop patterns and frequency
- Chart 5: Driver clustering visualization
- Chart 6: Race comparison analysis
- Chart 7: Feature correlation heatmap

---

## 🎓 Learning Outcomes

This project demonstrates:

- ✅ **Data Engineering** - API data collection, cleaning, preprocessing
- ✅ **Machine Learning** - Regression, classification, clustering
- ✅ **Model Evaluation** - Comprehensive metrics and comparison
- ✅ **Data Visualization** - Publication-ready charts
- ✅ **End-to-End Pipeline** - Complete data science workflow

---

## 📄 License

This project is open source and available under the MIT License.

---

## 🔗 Resources

- **FastF1 Documentation**: https://theoehrly.github.io/FastF1/
- **Scikit-learn Documentation**: https://scikit-learn.org/
- **XGBoost Documentation**: https://xgboost.readthedocs.io/

---

**Ready to analyze F1 race data? Open `F1_Analysis_Project.ipynb` in Jupyter!** 🏎️💨
