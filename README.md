# Sentinel-2 Image Classification Demo

A minimal Python project demonstrating supervised classification of Sentinel-2A Level-2A imagery using scikit-learn and rasterio.

## Features

- Load Sentinel-2 images with rasterio
- Prepare datasets for supervised classification
- Flexible sklearn classifier interface (easily switch between algorithms)
- Model persistence with pickle
- Generate classified raster outputs
- Simple visualization

## Project Structure

```
src/sentinel2_classifier/
├── data_loader.py      # Image loading and feature preparation
├── classifier.py       # Sklearn wrapper with model persistence
└── raster_processor.py # Output generation and visualization

train_model.py          # Training script
predict_image.py        # Prediction script for new images
```

## Usage

### 1. Train a Model

```bash
# Edit train_model.py to set your Sentinel-2 image path
python train_model.py
```

### 2. Classify New Images

```bash
# Edit predict_image.py to set input image path
python predict_image.py
```

### 3. Switch Classifiers

In `train_model.py`, easily change the classifier:

```python
# Random Forest (default)
classifier = Sentinel2Classifier(RandomForestClassifier(n_estimators=100))

# Support Vector Machine
classifier = Sentinel2Classifier(SVC(kernel='rbf'))

# Gradient Boosting
classifier = Sentinel2Classifier(GradientBoostingClassifier())
```

### index 

1. indices.py - Calculates NDVI and NDWI from Sentinel-2 bands:
   • NDVI: (NIR - Red) / (NIR + Red) - identifies vegetation
   • NDWI: (Green - NIR) / (Green + NIR) - identifies water

2. create_sample_labels_from_index() - Generates realistic labels using thresholds:
   • Water: NDWI > 0.3
   • Vegetation: NDVI > 0.4 and NDWI ≤ 0.3  
   • Urban: NDVI ≤ 0.4 and NDWI ≤ 0.3


## Input Requirements

- Sentinel-2A Level-2A processed images
- Multi-band GeoTIFF format
- Surface reflectance data (BOA - Bottom of Atmosphere)

## Output

- Classified GeoTIFF raster
- Visualization PNG
- Trained model pickle file

## Classes (Demo)

- 0: Water
- 1: Vegetation  
- 2: Urban

*Note: This demo uses synthetic labels. For real applications, use ground truth data or manual labeling.*


Pipeline Steps:
1. Data Loading - Load and inspect Sentinel-2 images
2. Index Calculation - Compute NDVI/NDWI and visualize
3. Dataset Generation - Create sklearn-compatible features and index-based labels
4. Train-Test Split - Proper dataset splitting with stratification
5. Model Training - Train RandomForest classifier
6. Evaluation - Classification report and confusion matrix
7. Full Classification - Classify entire image and visualize
8. Model Persistence - Save/load trained model with pickle
9. Output Generation - Save classified raster

Key Features:
• Uses all package functionality from src/
• Handles missing data with dummy generation for demo
• Includes proper evaluation metrics
• Visualizes indices, confusion matrix, and results
• Tests model loading to verify persistence

Run with:
bash
cd sentinel2-classifier
uv run jupyter notebook sentinel2_classification_pipeline.ipynb


The notebook provides a complete end-to-end workflow while leveraging all the modular components you've built.


Added:
• **Ruff** as dev dependency with uv
• **VSCode settings** for automatic formatting and linting on save
• **Ruff configuration** in pyproject.toml with Python 3.12 target
• **Makefile** with convenient commands

Configuration:
• Line length: 88 characters
• Import sorting and organization
• Error checking (E, F, I, N, W rules)
• Double quotes, space indentation

Usage:
bash
# Format code
make format
# or: uv run ruff format .

# Check for issues
make lint
# or: uv run ruff check .

# Fix issues automatically
make lint-fix
# or: uv run ruff check --fix .

# Format and fix everything
make check


VSCode Integration:
• Auto-format on save
• Import organization
• Real-time linting
• Uses project's virtual environment

All existing code has been formatted and linting issues fixed automatically.
