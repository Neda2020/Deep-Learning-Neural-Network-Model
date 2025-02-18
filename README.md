# 🧠 Deep Learning Neural Network Model  

## 📌 Project Overview  
This repository contains a **deep learning model** designed to predict the success of organizations receiving funding from **Alphabet Soup**. The model is a **binary classifier** built using **TensorFlow & Keras**, aiming to identify which applicants are most likely to utilize funding effectively.  

The project includes:  
✔️ **Data Preprocessing**: Handling categorical and numerical features.  
✔️ **Neural Network Model**: Defining architecture, activation functions, and layers.  
✔️ **Model Optimization**: Tuning hyperparameters for improved accuracy.  
✔️ **Evaluation & Insights**: Analyzing model performance and exploring alternative approaches.  

---

## 📂 Repository Structure  

```bash
📦 Deep-Learning-Neural-Network-Model-Public
│── 📜 LICENSE               # License information  
│── 📜 README.md             # Project documentation  
│── 📜 Report of Analysis.md # Detailed analysis report  
│── 📊 Charity_Optimization.ipynb   # Jupyter Notebook with full implementation  
│── 📊 charity.ipynb                # Alternative version of model implementation  
│── 🔢 Charity_Optimization.keras    # Trained model file  
│── 🔢 Charity_Optimization.h5       # Saved model weights  
│── 🔢 charity.keras                 # Another trained model version  
│── 🔢 Charity.h5                     # Additional model checkpoint  
🚀 Getting Started
📦 Prerequisites
To run this project, you need:

Python 3.7+
TensorFlow & Keras
Scikit-learn & Pandas
Jupyter Notebook (optional)
📥 Installation
Clone this repository and install the required dependencies:

bash
Copy
Edit
git clone https://github.com/Neda2020/Deep-Learning-Neural-Network-Model.git
cd Deep-Learning-Neural-Network-Model
pip install -r requirements.txt
📊 Model Architecture
The neural network consists of:

Input Layer: 36 input features
Hidden Layers:
First Layer: 80 neurons (ReLU activation)
Second Layer: 30 neurons (ReLU activation)
Output Layer: 1 neuron (Sigmoid activation)
Loss Function: binary_crossentropy
Optimizer: Adam
Epochs: 100+

📈 Model Performance
Metric	Value
Loss	0.5534
Accuracy	72.59%
🔹 The model achieved 72.59% accuracy, though further optimization is possible.
🔹 Alternative approaches (e.g., XGBoost, Random Forest) can be considered for improved performance.

⚙️ How to Use
1️⃣ Run the Jupyter Notebook
Open Charity_Optimization.ipynb and execute the cells to preprocess data, train the model, and evaluate results.

2️⃣ Load the Trained Model
If you want to use the pre-trained model, simply load it as follows:

python
Copy
Edit
from tensorflow.keras.models import load_model

model = load_model("Charity_Optimization.h5")
3️⃣ Make Predictions
Once loaded, you can make predictions:

python
Copy
Edit
prediction = model.predict(new_data)
📌 Future Enhancements
🔹 Improve accuracy using hyperparameter tuning.
🔹 Experiment with different neural network architectures.
🔹 Compare performance with traditional machine learning models.

📜 License
This project is licensed under the MIT License. See LICENSE for details.

