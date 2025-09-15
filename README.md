# 🚀 Bidirectional LSTM on IMDB Sentiment Analysis

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-2.x-red?logo=keras)
![NLP](https://img.shields.io/badge/NLP-IMDB-green?logo=google)
![License](https://img.shields.io/badge/License-MIT-brightgreen)

---

## 📚 Overview

This repository demonstrates a **Bidirectional LSTM** model for sentiment analysis on the IMDB movie reviews dataset. The project leverages deep learning techniques to classify reviews as positive or negative, providing a robust baseline for text classification tasks.

---

## 🛠️ Features

- **Bidirectional LSTM Architecture**: Captures context from both directions in text sequences.
- **Dropout Regularization**: Prevents overfitting and improves generalization.
- **Data Preprocessing**: Handles tokenization and padding for variable-length reviews.
- **Training & Validation Visualization**: Plots loss and accuracy curves for model evaluation.
- **Model Architecture Visualization**: Generates a flowchart of the neural network structure.
- **Easy Customization**: Modify hyperparameters and layers for experimentation.

---

## 📦 Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/Bidirectional-LSTM-IMDB.git
cd Bidirectional-LSTM-IMDB
```

### 2️⃣ Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn transformers datasets tensorflow keras pydot graphviz
```

> **Note:**  
> For model visualization, ensure [Graphviz](https://graphviz.gitlab.io/download/) is installed on your system.

---

## 📝 Usage

### 1️⃣ Run the Notebook

Open `main.ipynb` in **VS Code** or **Jupyter Notebook** and execute the cells step by step.

### 2️⃣ Model Training

- The model trains on the IMDB dataset for 10 epochs.
- Training and validation metrics are displayed after each epoch.

### 3️⃣ Visualization

- **Model Architecture**:  
  ![Model Architecture](https://img.shields.io/badge/Model-Architecture-blue?logo=graphviz)
  The flowchart is saved as `model.png` after running the visualization cell.
- **Training Curves**:  
  ![Training Curves](https://img.shields.io/badge/Training-Visualization-green?logo=plotly)
  Loss and accuracy plots are generated for performance analysis.

---

## 🔍 Project Structure

```
Bidirectional-LSTM-IMDB/
│
├── main.ipynb           # Main Jupyter Notebook
├── model.png            # Model architecture flowchart (generated)
├── README.md            # Project documentation
└── requirements.txt     # List of dependencies
```

---

## ⚙️ Model Details

| Property         | Description                                                                 |
|------------------|-----------------------------------------------------------------------------|
| **Embedding**    | Converts word indices to dense vectors (size: 128)                          |
| **Bidirectional LSTM** | Two layers, each with 64 units, capturing forward and backward context |
| **Dropout**      | 0.5 rate after each LSTM layer to reduce overfitting                        |
| **Dense Layer**  | 64 units with ReLU activation for feature extraction                        |
| **Output Layer** | 1 unit with Sigmoid activation for binary classification                    |

---

## 📊 Results

- **Training Accuracy**: Improves with epochs, indicating effective learning.
- **Validation Accuracy**: May plateau or decrease due to overfitting; regularization is applied.
- **Loss Curves**: Visualized for both training and validation sets.

---

## 💡 Tips for Improvement

- Tune hyperparameters (LSTM units, dropout rate, batch size).
- Experiment with additional regularization (L2, more dropout).
- Try different optimizers or learning rates.
- Use early stopping to prevent overfitting.

---

## 🖼️ Example Visualizations

<p align="center">
  <img src="model.png" alt="Model Architecture" width="500"/>
</p>

---

## 🤝 Contributing

Pull requests and suggestions are welcome!  
Please open an issue for feedback or feature requests.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## ✨ Acknowledgements

- [TensorFlow](https://www.tensorflow.org/)
- [Keras](https://keras.io/)