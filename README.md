# 👕 Fashion-MNIST Image Classification

A Deep Learning project that uses a **Convolutional Neural Network (CNN)** to classify Fashion-MNIST images into 10 different clothing categories.

The model is trained on **60,000 training images** and evaluated on **10,000 test images**, achieving approximately **92%+ test accuracy**.

## 🚀 Project Overview

Fashion-MNIST is a dataset of grayscale images representing different types of clothing and fashion products.

The objective of this project is to build a CNN capable of automatically identifying the category of a given clothing image.

## 📊 Dataset

The Fashion-MNIST dataset contains:

* **60,000** training images
* **10,000** test images
* Image size: **28 × 28 pixels**
* Grayscale images
* **10 classes**

### Classes

| Label | Category      |
| ----- | ------------- |
| 0     | T-shirt / Top |
| 1     | Trouser       |
| 2     | Pullover      |
| 3     | Dress         |
| 4     | Coat          |
| 5     | Sandal        |
| 6     | Shirt         |
| 7     | Sneaker       |
| 8     | Bag           |
| 9     | Ankle Boot    |

## 🧠 Model

A **Convolutional Neural Network (CNN)** is used for image classification.

The general workflow is:

```text
Fashion-MNIST Dataset
        ↓
Data Preprocessing
        ↓
Image Normalization
        ↓
CNN Model
        ↓
Model Training
        ↓
Model Evaluation
        ↓
Prediction
```

## 📈 Results

The CNN achieves approximately:

**92%+ Test Accuracy**

The model can predict the clothing category from unseen Fashion-MNIST images.

## 🛠️ Technologies Used

* Python
* TensorFlow
* Keras
* NumPy
* Matplotlib
* Jupyter Notebook

## 📂 Project Structure

```text
Fashion-Mnist/
│
├── api/
│   └── index.py
│
├── Fashion-Mnist.ipynb
├── train.py
├── requirements.txt
├── vercel.json
├── .gitignore
└── README.md
```

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/Rj-Ahsan/Fashion-Mnist.git
cd Fashion-Mnist
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## ▶️ Run the Project

Open the Jupyter Notebook:

```bash
jupyter notebook Fashion-Mnist.ipynb
```

You can also train the model using:

```bash
python train.py
```

## 🌐 Deployment

The repository includes a **Vercel serverless API setup** for model inference.

The API accepts a Fashion-MNIST image and returns the predicted classes and probabilities.

> **Note:** TensorFlow model training is intended to be performed locally. The Vercel setup is designed primarily for inference/deployment.

## 🔮 Future Improvements

* Improve CNN architecture
* Add data augmentation
* Experiment with transfer learning
* Add a web-based prediction interface
* Improve deployment architecture
* Add model performance visualizations

## 👨‍💻 Author

**Ahsan Tanveer**

BS Artificial Intelligence | AI/ML Engineer

GitHub: [@Rj-Ahsan](https://github.com/Rj-Ahsan)

---

⭐ If you find this project useful, consider giving the repository a star!
