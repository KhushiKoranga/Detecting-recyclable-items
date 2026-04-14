# Detecting-recyclable-items
AI-based web application to classify waste materials (glass, plastic, metal, paper, trash) and determine recyclability using MobileNetV2 and Streamlit.
# ♻️ Smart Waste Classifier

An AI-powered web application that classifies waste materials and determines whether they are recyclable or not using Deep Learning.

---

## 🚀 Features

* 📤 Upload image for prediction
* 🎥 Live camera detection
* 🧠 Material classification (glass, metal, paper, plastic, trash)
* ♻️ Recyclability detection
* 📊 Confidence score display

---

## 🧠 Technologies Used

* Python
* TensorFlow / Keras
* MobileNetV2 (Transfer Learning)
* Streamlit
* OpenCV
* NumPy

---

## 📁 Dataset Structure

```
dataset/
 ├── glass/
 ├── metal/
 ├── paper/
 ├── plastic/
 ├── trash/
```

---

## ⚙️ How It Works

1. Images are loaded and preprocessed (resize, normalize)
2. MobileNetV2 extracts features from images
3. Dense layers classify material type
4. Rule-based logic determines recyclability:

   * Trash → Not Recyclable ❌
   * Others → Recyclable ♻️

---

## 🧠 Model Details

* Transfer Learning using MobileNetV2
* Input size: 224×224
* Activation: ReLU & Softmax
* Optimizer: Adam
* Loss: Sparse Categorical Crossentropy

---

## ▶️ How to Run

### Step 1: Install dependencies

```
pip install -r requirements.txt
```

### Step 2: Train model

```
python model/train.py
```

### Step 3: Run app

```
python -m streamlit run app.py
```

---

## ⚠️ Limitations

* Cannot perfectly detect recyclability (depends on cleanliness)
* May confuse similar materials
* Accuracy depends on dataset quality

---

## 🔮 Future Improvements

* Use object detection (YOLO)
* Improve dataset diversity
* Deploy as a mobile/web application

---

## 👩‍💻 Author

Khushi Koranga
B.Tech AI/ML Student

---

## ⭐ Conclusion

This project demonstrates how AI can be used to assist in waste management and promote environmental sustainability.
