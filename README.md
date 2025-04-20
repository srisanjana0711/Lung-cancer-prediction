# Lung Cancer Prediction using CNN and Ensemble Learning

This project aims to detect lung cancer using image classification techniques based on **Convolutional Neural Networks (CNN)** combined with **ensemble learning** approaches such as **Random Forest**, **Stacking**, **Voting**, and **Weighted Average**. It includes preprocessing steps like noise reduction and contrast enhancement, feature extraction using CNN, and visualization of performance metrics.

---

## 📁 Project Structure

- `lung.py` - The main Python script with full pipeline from data preprocessing to model training and evaluation.
- `/Dset/` - Contains images categorized into:
  - `Benign Cases`
  - `Malignant Cases`
  - `Normal Cases`
- `dset_labels.csv` - CSV file with `Image_Path` and `Label` columns.
- Saved models and result plots are stored in Google Drive (`/content/drive/MyDrive/Lung/`).

---

## 🔍 Problem Statement

Early detection of lung cancer is vital to increase survival rates. This project leverages deep learning and machine learning models to classify lung CT scan images into **cancerous (Benign or Malignant)** and **non-cancerous (Normal)**.

---

## 🚀 Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- Scikit-learn
- Matplotlib / Seaborn
- Google Colab

---

## ⚙️ Model Pipeline

1. **Data Preprocessing**
   - Path updates from CSV
   - Image resizing to 224×224
   - Gaussian Blur for noise reduction
   - CLAHE for contrast enhancement
   - Normalization to [0, 1]

2. **CNN Model**
   - 3 Conv2D layers with MaxPooling
   - Fully connected Dense layer with Dropout
   - Binary classification with sigmoid activation

3. **Feature Extraction**
   - Features are extracted from CNN's penultimate dense layer

4. **Random Forest Classifier**
   - Trained using extracted features
   - Accuracy, precision, recall, F1-score evaluation

5. **Ensemble Techniques**
   - **Stacking**: Combines Random Forest & SVM with Logistic Regression as meta-classifier
   - **Voting**: Majority voting between base learners
   - **Weighted Average**: Weighted soft voting using model performance

6. **Visualization**
   - Confusion Matrix
   - Classification Report (Heatmap)
   - ROC Curve
   - Gini Index Plot

---

## 📊 Results

- CNN Accuracy: ~95%
- Random Forest Accuracy: ~95%
- Ensemble (Stacking/Voting/Weighted): Improved performance
- ROC AUC: Displayed with ROC Curve

---

## 🧠 Key Learnings

- Integrating deep learning (CNN) and machine learning (RF, SVM) improves classification accuracy.
- Ensemble models provide better generalization.
- Visualization tools help understand model performance and fairness.

---

## ✅ How to Run

1. Clone the repo and upload to Google Colab.
2. Mount your Google Drive with `drive.mount('/content/drive')`.
3. Ensure dataset and `dset_labels.csv` are placed in `/MyDrive/Lung/Dset/`.
4. Run the script cell by cell.
5. Outputs like models and plots will be saved to your Drive.

---

## 📌 Note

- Make sure your dataset is labeled as:  
  - `1` → Cancerous (Benign or Malignant)  
  - `0` → Normal
- The project is designed to work seamlessly in Google Colab.

---



