# Land Type Classification using Sentinel-2 Satellite Images

![Project Overview](eurosat_overview_small.jpg)

## Overview
This project focuses on leveraging **Deep Neural Networks (DNNs)** to classify different land types (such as **agriculture, water, urban areas, desert, roads, and trees**) using satellite imagery from the **Sentinel-2** mission by the European Space Agency (ESA). The goal is to build an accurate classification model that can be useful for applications like **urban planning, environmental monitoring, and resource management**.

## Dataset
- **Source**: Sentinel-2 satellite images
- **Alternative Dataset**: [EuroSAT Dataset](https://github.com/phelber/EuroSAT)
- **Image Type**: Multispectral images with different spectral bands (Red, Green, Blue, Near-Infrared, etc.)
- **Preprocessing**: Resized, normalized, and enhanced using various techniques (e.g., atmospheric correction)

## Project Milestones

### 1. Data Collection, Exploration & Preprocessing
- **Download Sentinel-2 images** for the target region (e.g., Egypt) from [Copernicus Open Access Hub](https://scihub.copernicus.eu/dhus/#/home)
- **Perform Exploratory Data Analysis (EDA)** to understand image composition and band relevance
- **Preprocess the dataset** (resizing, band selection, image augmentation, and NDVI calculation)
- **Visualize spectral signatures** of different land types

### 2. Advanced Data Analysis & Model Selection
- Identify **key spectral bands** that influence land classification
- Perform **Principal Component Analysis (PCA)** to reduce dimensionality
- Experiment with various **Deep Learning models**, starting with **CNNs (Convolutional Neural Networks)**
- Consider **transfer learning** using pre-trained models (e.g., ResNet, VGG, or U-Net)

### 3. Model Development & Training
- Implement **CNN-based DNN models** using **TensorFlow/Keras or PyTorch**
- Train the model with **early stopping and hyperparameter tuning**
- Evaluate performance using metrics like **accuracy, precision, recall, and F1-score**
- Visualize results using **confusion matrices and activation maps**

### 4. Deployment & Monitoring
- Deploy the model as a **web service/API** using **Flask or FastAPI**
- Monitor classification accuracy and detect **model drift** over time
- Implement a strategy for **periodic retraining** with new satellite images

## Installation & Usage
### Requirements
- Python 3.8+
- TensorFlow / PyTorch
- NumPy, Pandas, Matplotlib, Seaborn
- OpenCV, Rasterio (for image processing)
- Flask / FastAPI (for deployment)

### Setup
```bash
# Clone the repository
git clone https://github.com/your-repo/land-type-classification.git
cd land-type-classification

# Install dependencies
pip install -r requirements.txt
```

### Running the Model
```bash
python train_model.py
```

### Deploying the Model
```bash
python app.py
```

## Results & Visualizations
- **Confusion Matrix** to evaluate model performance
- **NDVI & Spectral Band Analysis** for land type differentiation
- **Web Interface / API** to classify uploaded images

## Future Improvements
- Incorporating **time-series analysis** for seasonal land changes
- Enhancing model accuracy with **higher-resolution datasets**
- Developing a **real-time dashboard** for land classification monitoring

---
**Author:** [Team 6]  
**Contact:** soadatef199@gmail.com.com  
**License:** MIT License

