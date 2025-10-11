# Sentinel-2 Image Classification Demo

A minimal Python project demonstrating supervised classification of Sentinel-2A Level-2A imagery using scikit-learn and rasterio.

## Features

- Load Sentinel-2 images with rasterio
- Prepare datasets for supervised classification
- Flexible sklearn classifier interface (easily switch between algorithms)
- Model persistence with pickle
- Generate classified raster outputs
- Simple visualization
- **Robust logging system with configurable levels**

## Project Structure

```
src/sentinel2_classifier/
├── data_loader.py      # Image loading and feature preparation
├── classifier.py       # Sklearn wrapper with model persistence
├── raster_processor.py # Output generation and visualization
├── logging_config.py   # Logging configuration and setup
└── ...

train_model.py          # Training script
predict_image.py        # Prediction script for new images
logging_config.py       # Logging configuration example
```

## Logging Configuration

The project uses a comprehensive logging system that replaces all print statements:

### Default Configuration
- **Default level**: INFO
- **Format**: `%(asctime)s - %(name)s - %(levelname)s - %(message)s`
- **Output**: stdout

### Control Logging Level

Set environment variable to control logging globally:
```bash
# Debug level (most verbose)
export SENTINEL2_LOG_LEVEL=DEBUG

# Info level (default)
export SENTINEL2_LOG_LEVEL=INFO

# Warning level
export SENTINEL2_LOG_LEVEL=WARNING

# Error level (least verbose)
export SENTINEL2_LOG_LEVEL=ERROR
```

### Logging Levels Used
- **INFO**: Step information, progress updates, results
- **DEBUG**: Detailed execution information, data shapes, parameters
- **WARNING**: Non-critical issues, fallback behaviors
- **ERROR**: Critical errors, file not found, processing failures

### Example Usage
```python
from src.sentinel2_classifier import setup_logger

# Setup custom logger
logger = setup_logger("my_script", level="DEBUG")

# Use logger instead of print
logger.info("Processing started")
logger.debug(f"Data shape: {data.shape}")
logger.warning("Using fallback method")
logger.error("Processing failed")
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

### 4. Control Logging Output

```bash
# Run with debug logging
SENTINEL2_LOG_LEVEL=DEBUG python process_multispectral.py

# Run with minimal logging
SENTINEL2_LOG_LEVEL=ERROR python train_model.py
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

New Features:

1. geospatial_utils.py - Handles GeoJSON processing and CRS transformations:
   • Loads GeoJSON files
   • Validates and transforms CRS (EPSG:3857 ↔ EPSG:4326)
   • Crops raster data using polygon geometries
   • Extracts bounding boxes

2. Updated resampling.py - Now supports optional GeoJSON cropping:
   • Added geojson_path parameter to resampling functions
   • Integrates cropping after band resampling
   • Maintains geospatial referencing

3. Updated load_sentinel2_multispectral() - Enhanced with cropping capability:
   • Optional geojson_path parameter
   • Processes full 13-band Sentinel-2 data
   • Resamples all bands to common resolution (10m, 20m, or 60m)
   • Crops to region of interest if GeoJSON provided

4. Updated process_multispectral.py - Demonstrates the full workflow:
   • References data/cdmx.json for CDMX region cropping
   • Outputs different files for cropped vs full processing

Usage:
python
# Full image processing
data, profile, bands = load_sentinel2_multispectral(safe_folder, target_resolution=10)

# Cropped processing with GeoJSON
data, profile, bands = load_sentinel2_multispectral(
    safe_folder, target_resolution=10, geojson_path="data/cdmx.json"
)


The system now handles the complete Sentinel-2 multispectral workflow with proper band resampling and optional region-of-interest cropping using 
your existing GeoJSON definition.

Key Changes:

1. resample_sentinel2_bands() - Now filters bands to only use those at the target resolution:
   • No more resampling between different resolutions
   • Avoids matrix size conflicts (10980x10980 vs 5490x5490 vs 1830x1830)
   • Only loads bands that naturally exist at the target resolution

2. load_sentinel2_safe_folder() - Updated with resolution-specific band mapping:
   • **10m bands**: B02, B03, B04, B08 (RGB + NIR)
   • **20m bands**: B05, B06, B07, B8A, B11, B12 (Red Edge + SWIR)
   • **60m bands**: B01, B09, B10 (Coastal + Water Vapor + Cirrus)

3. Automatic band selection - Based on target resolution:
   • target_resolution=10 → Uses only 10m bands
   • target_resolution=20 → Uses only 20m bands  
   • target_resolution=60 → Uses only 60m bands

Usage:
python
# Load 10m resolution data (B02, B03, B04, B08)
data, profile, bands = load_sentinel2_multispectral(safe_folder, target_resolution=10)

# Load 20m resolution data (B05, B06, B07, B8A, B11, B12)  
data, profile, bands = load_sentinel2_multispectral(safe_folder, target_resolution=20)


This approach eliminates resampling complexity while ensuring all bands have consistent dimensions for the sklearn dataset creation.


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
