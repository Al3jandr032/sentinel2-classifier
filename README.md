# 🛰️ Sentinel-2 Image Classification Demo

> **Python Meetup Demo**: Classify satellite imagery using scikit-learn and rasterio

Transform Sentinel-2 satellite images into land cover maps with just a few lines of Python code. Perfect for demonstrating the power of Python in geospatial data science!

## 🚀 Quick Start

```bash
# Clone and setup
git clone <your-repo>
cd sentinel2-classifier
uv sync

# Run the complete demo
uv run jupyter notebook sentinel2_classification_pipeline.ipynb
```

## ✨ What This Demo Shows

- **Load satellite imagery** with rasterio
- **Calculate vegetation indices** (NDVI, NDWI)
- **Train ML models** with scikit-learn
- **Generate land cover maps** automatically
- **Visualize results** with matplotlib

## 🎯 Demo Results

The classifier identifies three land cover types:
- 🌊 **Water** (rivers, lakes)
- 🌱 **Vegetation** (forests, crops)
- 🏙️ **Urban** (buildings, roads)

## 📁 Project Structure

```
src/sentinel2_classifier/
├── data_loader.py      # Load Sentinel-2 images
├── classifier.py       # ML model wrapper
├── indices.py          # NDVI/NDWI calculation
├── raster_processor.py # Generate outputs
└── geospatial_utils.py # GeoJSON cropping

sentinel2_classification_pipeline.ipynb  # 📓 Main demo notebook
```

## 🔧 Key Features

**Smart Band Selection**
```python
# Automatically uses bands at target resolution
data, profile, bands = load_sentinel2_multispectral(
    safe_folder, target_resolution=10  # Uses B02, B03, B04, B08
)
```

**Flexible Classifiers**
```python
# Easy to switch algorithms
classifier = Sentinel2Classifier(RandomForestClassifier())
classifier = Sentinel2Classifier(SVC(kernel='rbf'))
```

**Region Cropping**
```python
# Focus on specific areas with GeoJSON
data, profile, bands = load_sentinel2_multispectral(
    safe_folder, geojson_path="data/cdmx.json"
)
```

## 🎪 Live Demo Flow

1. **Load Data** → Sentinel-2 multispectral image
2. **Calculate Indices** → NDVI for vegetation, NDWI for water
3. **Generate Labels** → Automatic thresholding (demo only!)
4. **Train Model** → RandomForest classifier
5. **Classify Image** → Full scene prediction
6. **Visualize Results** → Beautiful land cover map

## 🛠️ Development

```bash
# Format code
make format

# Run linting
make lint

# Fix issues
make lint-fix
```

## 📊 Input Data

- **Sentinel-2A Level-2A** (surface reflectance)
- **Multi-band GeoTIFF** format
- **Any resolution**: 10m, 20m, or 60m

## 💡 Perfect For Learning

- **Geospatial Python** fundamentals
- **Machine learning** with real data
- **Satellite imagery** processing
- **Scientific visualization**

---

*This is a demo project using synthetic labels. For production use, collect proper ground truth data!*
