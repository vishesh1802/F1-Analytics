# 🏎️ F1 Data Analysis Project

A comprehensive machine learning project analyzing Formula 1 race data to predict lap times, optimize pit stop strategy, and cluster driver performance.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Completed-success.svg)]()

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Results](#results)
- [Documentation](#documentation)
- [Technologies Used](#technologies-used)
- [Project Components](#project-components)
- [Contributing](#contributing)
- [License](#license)

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

## ✨ Features

### Data Collection
- Automated data download from FastF1 API
- Weather data integration (air & track temperature)
- Intelligent caching for faster subsequent runs
- Support for multiple races and seasons

### Machine Learning Models

**Regression Models:**
- Linear Regression (baseline)
- Random Forest Regressor
- XGBoost Regressor ⭐ (Best: R² = 0.92)

**Classification Models:**
- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier ⭐ (Best: 100% accuracy)

**Clustering Algorithms:**
- K-Means Clustering
- Hierarchical Clustering
- DBSCAN Clustering

### Comprehensive Evaluation
- Regression metrics: RMSE, MAE, R²
- Classification metrics: Accuracy, Precision, Recall, F1, AUC-ROC
- Clustering metrics: Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index

### Visualizations
- 7 comprehensive charts and visualizations
- Interactive data exploration tools
- Power BI dashboard ready datasets

---

## 📁 Project Structure

```
F1 Analysis/
│
├── 📄 README.md                          # This file
├── 📄 requirements.txt                    # Python dependencies
│
├── 🐍 f1_project.py                      # Main analysis script
├── 📓 F1_Analysis_Project.ipynb          # Jupyter notebook version
├── 📊 create_charts.py                   # Visualization script
├── 📊 powerbi_export.py                  # Power BI data export
│
├── 📊 Data Files/
│   ├── raw_laps.csv                      # Raw combined data (8,763 laps)
│   ├── laptime_dataset.csv               # Preprocessed lap-time data
│   ├── pit_dataset.csv                   # Preprocessed pit-stop data
│   ├── driver_stats.csv                  # Driver statistics
│   └── driver_clusters.csv               # Driver clusters with labels
│
├── 📈 charts/                             # Generated visualizations
│   ├── chart1_lap_time_distribution.png
│   ├── chart2_top_drivers.png
│   ├── chart3_temperature_effects.png
│   ├── chart4_pit_stop_patterns.png
│   ├── chart5_driver_clustering.png
│   ├── chart6_race_comparison.png
│   └── chart7_correlation_heatmap.png
│
├── 📚 Documentation/
│   ├── DETAILED_PROJECT_REPORT.md        # Complete project report
│   ├── PROJECT_EXPLANATION.md            # Project overview
│   ├── COLUMN_DETAILED_EXPLANATION.md    # Column reference guide
│   ├── POWERBI_DASHBOARD_GUIDE.md        # Power BI setup guide
│   ├── REQUIREMENTS_CHECKLIST.md         # Proposal requirements
│   ├── PROPOSAL_COMPARISON.md            # Requirements vs implementation
│   ├── ENHANCEMENTS_SUMMARY.md           # Added features
│   └── RESULTS_ANALYSIS.md                # Results interpretation
│
└── 💾 cache/                              # FastF1 cached data
    └── 2023/                              # Race data cache
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Internet connection (for initial data download)

### Step 1: Clone or Download

```bash
# If using git
git clone <repository-url>
cd "F1 Analysis"

# Or download and extract the project folder
```

### Step 2: Install Dependencies

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

### Step 3: Verify Installation

```bash
python -c "import fastf1, pandas, sklearn, xgboost; print('✅ All packages installed successfully!')"
```

---

## ⚡ Quick Start

### Option 1: Run Python Script

```bash
python f1_project.py
```

This will:
1. Download/load F1 race data (first run: ~5-15 min, subsequent: ~30 sec)
2. Preprocess data into 3 datasets
3. Train 9 machine learning models
4. Evaluate and compare all models
5. Generate CSV files with results

### Option 2: Use Jupyter Notebook

```bash
# Start Jupyter
jupyter notebook

# Open F1_Analysis_Project.ipynb
# Run all cells (Cell → Run All)
```

### Option 3: Generate Visualizations

```bash
python create_charts.py
```

This creates 7 charts in the `charts/` folder.

---

## 📖 Usage

### Basic Usage

**1. Run the main analysis:**
```bash
python f1_project.py
```

**2. View results:**
- Check console output for model metrics
- Open CSV files to explore data
- View charts in `charts/` folder

**3. Generate Power BI datasets:**
```bash
python powerbi_export.py
```

### Advanced Usage

#### Customize Races

Edit `f1_project.py` to change the race list:

```python
RACES = [
    (2023, "Bahrain"),
    (2023, "Monaco"),      # Add more races
    (2024, "Bahrain"),     # Different year
]
```

#### Modify Models

Edit model parameters in the respective functions:

```python
# Example: Change XGBoost parameters
pipeline_xgb = Pipeline([
    ('preprocessor', preprocessor),
    ('model', xgb.XGBRegressor(
        n_estimators=200,  # Increase trees
        max_depth=6,       # Add depth limit
        learning_rate=0.1  # Adjust learning rate
    ))
])
```

#### Custom Visualizations

Modify `create_charts.py` to create custom visualizations or add new charts.

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
| Algorithm | Silhouette | Davies-Bouldin | Calinski-Harabasz |
|-----------|------------|----------------|-------------------|
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

## 📚 Documentation

### Comprehensive Guides

- **[DETAILED_PROJECT_REPORT.md](DETAILED_PROJECT_REPORT.md)** - Complete academic-style report
- **[PROJECT_EXPLANATION.md](PROJECT_EXPLANATION.md)** - Detailed project explanation
- **[COLUMN_DETAILED_EXPLANATION.md](COLUMN_DETAILED_EXPLANATION.md)** - Every column explained
- **[POWERBI_DASHBOARD_GUIDE.md](POWERBI_DASHBOARD_GUIDE.md)** - Power BI setup instructions

### Quick References

- **[REQUIREMENTS_CHECKLIST.md](REQUIREMENTS_CHECKLIST.md)** - Proposal requirements status
- **[PROPOSAL_COMPARISON.md](PROPOSAL_COMPARISON.md)** - Requirements vs implementation
- **[ENHANCEMENTS_SUMMARY.md](ENHANCEMENTS_SUMMARY.md)** - Added features summary
- **[RESULTS_ANALYSIS.md](RESULTS_ANALYSIS.md)** - Results interpretation

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

## 🎯 Project Components

### 1. Data Collection (`load_raw_laps()`)
- Downloads race data from FastF1 API
- Merges weather data with lap data
- Caches data for faster subsequent runs
- Handles errors gracefully

### 2. Data Preprocessing
- **`make_laptime_dataset()`** - Creates regression dataset
- **`make_pit_dataset()`** - Creates classification dataset
- **`make_driver_stats()`** - Creates clustering dataset

### 3. Machine Learning Models
- **`run_laptime_model()`** - Trains 3 regression models
- **`run_pit_model()`** - Trains 3 classification models
- **`run_driver_clustering()`** - Applies 3 clustering algorithms

### 4. Visualization
- **`create_charts.py`** - Generates 7 comprehensive charts
- Interactive data exploration
- Publication-ready visualizations

### 5. Power BI Integration
- **`powerbi_export.py`** - Exports Power BI-ready datasets
- Includes predictions and model results
- Ready for dashboard creation

---

## 📈 Data Overview

### Dataset Statistics
- **Total Laps:** 8,763
- **Races:** 8 (2023 season)
- **Drivers:** 21
- **Features:** 34 columns per lap
- **Time Period:** 2023 Formula 1 season

### Races Analyzed
1. Bahrain Grand Prix
2. Saudi Arabian Grand Prix
3. Australian Grand Prix
4. Azerbaijan Grand Prix
5. Spanish Grand Prix
6. Canadian Grand Prix
7. Austrian Grand Prix
8. Belgian Grand Prix

---

## 🔧 Configuration

### Key Settings

Edit these in `f1_project.py`:

```python
# Cache directory for FastF1
CACHE_DIR = "cache"

# Races to analyze
RACES = [
    (2023, "Bahrain"),
    # Add more races here
]

# Export CSV files
SAVE_CSV = True
```

---

## 🐛 Troubleshooting

### Common Issues

**Issue: "Cache directory does not exist"**
```bash
# Solution: The script now creates it automatically
# If error persists, manually create:
mkdir cache
```

**Issue: "Import errors"**
```bash
# Solution: Reinstall dependencies
pip install --upgrade -r requirements.txt
```

**Issue: "Slow data download"**
```bash
# Solution: First run downloads data (5-15 min)
# Subsequent runs use cache (~30 sec)
# Data is cached in cache/ folder
```

**Issue: "Missing temperature columns"**
```bash
# Solution: Delete old CSV and regenerate
rm raw_laps.csv
python f1_project.py
```

---

## 📝 Example Output

```
============================================================
F1 DATA ANALYSIS PROJECT
============================================================
Loading data from cached CSV file: raw_laps.csv
Loaded 8763 rows, 34 columns from cache

============================================================
PREPROCESSING
============================================================
Lap-time dataset: 8568 rows
Pit-stop dataset: 8728 rows
Driver stats dataset: 21 drivers

============================================================
LAP-TIME PREDICTION MODEL (Regression)
============================================================
--- XGBoost Regressor ---
RMSE: 4.489 seconds
MAE:  1.995 seconds
R²:   0.9212

============================================================
PIT-STOP PREDICTION MODEL (Classification)
============================================================
--- Gradient Boosting Classifier ---
Accuracy:  1.000
AUC-ROC:   1.0000

============================================================
DRIVER CLUSTERING (Multiple Algorithms)
============================================================
--- K-Means Clustering ---
Silhouette Score: 0.4068
3 clusters identified

ANALYSIS COMPLETE!
```

---

## 🎓 Learning Outcomes

This project demonstrates:

- ✅ **Data Engineering** - API data collection, cleaning, preprocessing
- ✅ **Machine Learning** - Regression, classification, clustering
- ✅ **Model Evaluation** - Comprehensive metrics and comparison
- ✅ **Data Visualization** - Publication-ready charts
- ✅ **End-to-End Pipeline** - Complete data science workflow

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Additional features (sector times, position data)
- Advanced models (neural networks, time series)
- More visualizations
- Performance optimizations
- Documentation improvements

---

## 📄 License

This project is open source and available under the MIT License.

---

## 👤 Author

**F1 Data Analysis Project**
- Analysis of 2023 Formula 1 season
- Machine learning for race strategy optimization
- Comprehensive evaluation and visualization

---

## 🙏 Acknowledgments

- **FastF1** - For providing excellent F1 data API
- **Formula 1** - For making race data available
- **Scikit-learn** - For comprehensive ML tools
- **Open Source Community** - For amazing Python libraries

---

## 📞 Support

For questions or issues:
1. Check the documentation files
2. Review the code comments
3. Check FastF1 documentation: https://theoehrly.github.io/FastF1/

---

## 🎯 Next Steps

1. ✅ Run the analysis: `python f1_project.py`
2. ✅ Generate charts: `python create_charts.py`
3. ✅ Explore the data in CSV files
4. ✅ Create Power BI dashboard (see guide)
5. ✅ Read the detailed report for insights

---

## 📊 Project Status

- ✅ Data Collection - Complete
- ✅ Data Preprocessing - Complete
- ✅ Model Development - Complete
- ✅ Model Evaluation - Complete
- ✅ Visualization - Complete
- ✅ Documentation - Complete
- ⏳ Power BI Dashboard - Ready (see guide)

**Overall Completion: 95%** 🎉

---

## 🌟 Highlights

- 🏆 **Best-in-class performance** - 92% R² for lap time prediction
- 🎯 **Perfect classification** - 100% accuracy for pit stops
- 📊 **Comprehensive analysis** - 9 models, 3 problem types
- 📈 **Rich visualizations** - 7 publication-ready charts
- 📚 **Complete documentation** - Multiple detailed guides
- 🔄 **Reproducible** - All code and data included

---

**Ready to analyze F1 race data? Start with `python f1_project.py`!** 🏎️💨


