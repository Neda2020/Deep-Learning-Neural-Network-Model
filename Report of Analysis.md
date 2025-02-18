# Predicting Funding Success for Alphabet Soup: A Deep Learning Approach  

## Introduction  
Organizations applying for Alphabet Soup funding have varying levels of success in utilizing the financial support. The goal of this study is to develop a **binary classification model** using deep learning to predict which applicants are most likely to use the funding effectively.  

This analysis involves:  
✔️ **Data Preprocessing** (handling categorical and numerical data)  
✔️ **Neural Network Model Design** (defining architecture and parameters)  
✔️ **Training & Optimization** (tuning hyperparameters)  
✔️ **Evaluation & Improvement** (analyzing results and exploring alternatives)  

---

## Data Preparation  

### 🎯 **Target Variable**  
- `IS_SUCCESSFUL`: Indicates whether an organization effectively utilized the funding.  

### 🏛️ **Feature Variables**  
The following variables were used to build the model:  
- `APPLICATION_TYPE`, `AFFILIATION`, `CLASSIFICATION`, `USE_CASE`  
- `ORGANIZATION`, `STATUS`, `INCOME_AMT`, `SPECIAL_CONSIDERATIONS`, `ASK_AMT`  

### ❌ **Removed Columns**  
- `EIN` and `NAME` were dropped as they are only **identifiers** and **do not contribute to predictions**.  

### 🔄 **Data Preprocessing Steps**  
✔️ **Grouping Rare Categories:** Less frequent categorical values were merged into an **"Other"** category.  
✔️ **Encoding Categorical Variables:** Applied **one-hot encoding (`pd.get_dummies()`)** to convert them into numerical format.  
✔️ **Splitting Data:** The dataset was divided into **training and testing sets** using `train_test_split()`.  
✔️ **Scaling Features:** Used **StandardScaler** to normalize numerical values for better model efficiency.  

---

## Building the Neural Network  

### 🏗️ **Model Architecture**  
- **Input Layer:** 36 input features  
- **Hidden Layers:**  
  - Layer 1: **80 neurons**, ReLU activation  
  - Layer 2: **30 neurons**, ReLU activation  
- **Output Layer:** 1 neuron with **sigmoid activation** (for binary classification)  

### ⚙️ **Model Compilation & Training**  
- **Loss Function:** `binary_crossentropy` (suited for binary classification)  
- **Optimizer:** `Adam` (adaptive learning rate)  
- **Metric:** `accuracy`  
- **Training Epochs:** **100**  

---

## Model Evaluation  

### 📊 **Performance Metrics**  
- **Loss:** `0.5534`  
- **Accuracy:** **72.59%**  

### 🔎 **Observations**  
- The model **did not achieve the target accuracy of 75%**, despite adjustments.  
- Increasing neurons and adding another hidden layer **did not significantly improve performance**.  
- The dataset may already contain the **maximum predictive power possible** with the given features.  
- The model might be **overfitting**, despite similar test accuracy.  

---

## Improving the Model  

### ✅ **Modifications Attempted**  
- **Increased neuron count** per layer  
- **Added an additional hidden layer**  
- **Extended training to 120 epochs**  

### ❗ **Findings**  
🔹 Accuracy **remained at ~72%**, indicating **no major improvement** with added complexity.  
🔹 The model may require **better data rather than a deeper architecture**.  
🔹 **Feature Engineering** could provide a better path for improving accuracy.  

### 🔧 **Potential Enhancements**  
- **Fine-tune hyperparameters** using Grid Search or Keras Tuner  
- **Enhance preprocessing** by improving feature selection and scaling  
- **Reduce overfitting** by applying regularization techniques  

---

## Alternative Models  

Since deep learning did not significantly enhance accuracy, alternative models should be explored:  

### 🌲 **Tree-Based Models (Random Forest, XGBoost)**  
✔️ **Handles categorical variables better**  
✔️ **Can capture complex, non-linear relationships**  
✔️ **Provides feature importance insights**  

### 🔺 **Support Vector Machine (SVM)**  
✔️ **RBF Kernel SVM** can be effective for datasets with **overlapping classes**.  

---

## Conclusion  

📌 **Key Takeaways:**  
- The neural network **achieved 72.59% accuracy**, but could not surpass **75%**.  
- Increasing model complexity **did not yield improvements**, suggesting **data limitations**.  
- **Alternative models** (e.g., XGBoost, SVM) may be more effective.  

📌 **Next Steps:**  
🔹 Focus on **better data preprocessing & feature engineering**.  
🔹 Optimize the model using **hyperparameter tuning**.  
🔹 Consider **alternative machine learning approaches** for better performance.  

🚀 **Further refinements and model selection will be critical to achieving higher accuracy.**
