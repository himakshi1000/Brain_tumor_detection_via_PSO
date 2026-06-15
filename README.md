---

# 🧠 CNN Image Classification with PSO Hyperparameter Optimization

This project implements a **Convolutional Neural Network (CNN)** for image classification, with hyperparameters optimized using **Particle Swarm Optimization (PSO)**. The workflow is designed in Google Colab and leverages TensorFlow/Keras, OpenCV, and Scikit-learn.

---

## 📂 Project Structure

- **Training Data**: `/content/drive/MyDrive/miniproject_5thsem/Training`
- **Testing Data**: `/content/drive/MyDrive/miniproject_5thsem/Testing`
- **Notebook**: Contains data loading, preprocessing, CNN model creation, PSO optimization, and evaluation.

---

## ⚙️ Dependencies

The following libraries are required:

- TensorFlow / Keras
- OpenCV
- NumPy
- Scikit-learn
- Seaborn
- Matplotlib
- Pandas
- PySwarm

Install PySwarm:
```bash
pip install pyswarm
```

---

## 📊 Data Preparation

- Images are loaded from class-specific directories.
- Each image is resized to **64x64 pixels** and normalized to `[0,1]`.
- Labels are converted to **categorical format** for multi-class classification.

```python
img = cv2.imread(img_path, cv2.IMREAD_COLOR)
img = cv2.resize(img, (64, 64))
X_train = X_train.astype('float32') / 255.0
```

---

## 🏗️ Model Architecture

The CNN model is defined as:

- **Conv2D** layer with ReLU activation
- **MaxPooling2D**
- **Flatten**
- **Dense** layer with ReLU
- **Dropout**
- **Output Dense** layer with Softmax

Optimizer: **Adam** with variable learning rate.

```python
model.add(Conv2D(num_filters, (3,3), activation='relu', input_shape=(64,64,3)))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(dense_units, activation='relu'))
model.add(Dropout(dropout_rate))
model.add(Dense(num_classes, activation='softmax'))
```

---

## 🌀 Hyperparameter Optimization with PSO

PSO is used to optimize:

- Learning Rate
- Number of Filters
- Dense Units
- Dropout Rate
- Batch Size

Bounds:
```python
bounds = [
    (1e-5, 1e-2),  # learning_rate
    (16, 64),      # num_filters
    (64, 256),     # dense_units
    (0.1, 0.5),    # dropout_rate
    (16, 64)       # batch_size
]
```

Objective: **maximize accuracy** (minimize negative accuracy).

---

## 📈 Training & Evaluation

- Early stopping is applied to prevent overfitting.
- Models are trained for **10 epochs** with validation split.
- Final evaluation is performed on the test set.

Metrics:
- Accuracy
- Confusion Matrix
- Classification Report
- ROC-AUC (optional)

---

## 🚀 Results

- PSO finds the best hyperparameters automatically.
- Final model is retrained with optimal parameters.
- Test accuracy is reported at the end.

Example output:
```
Best hyperparameters found by PSO:
Learning Rate: 0.0012
Number of Filters: 32
Dense Units: 128
Dropout Rate: 0.3
Batch Size: 32

Final test accuracy: 0.56
```

---

## 📌 Key Features

- Automated hyperparameter tuning with **PSO**
- Simple yet effective **CNN architecture**
- Integration with **Google Drive** for dataset management
- Visualization with **Seaborn** and **Matplotlib**

---

## 🔮 Future Improvements

- Add more convolutional layers for deeper feature extraction
- Use **data augmentation** to improve generalization
- Experiment with **transfer learning** using pretrained models
- Extend PSO iterations and swarm size for better optimization

---
