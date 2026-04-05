# 📊 Customer Churn Prediction using ANN

This project focuses on predicting customer churn using an **Artificial Neural Network (ANN)**. It helps identify whether a customer is likely to leave a service based on various features like credit score, geography, age, balance, etc.

---

## 🚀 Project Overview

Customer churn is a critical problem for businesses. This project uses **Deep Learning** to analyze customer data and predict churn probability, 
enabling companies to take proactive actions.

---

## 🧠 Model Used

- Artificial Neural Network (ANN)
- Built using TensorFlow/Keras
- Binary Classification Problem

---

## 📂 Project Structure

```
├── app.py                     # Streamlit web app
├── experiments.ipynb          # Model training & experimentation
├── prediction.ipynb           # Prediction testing
├── Churn_Modelling.csv        # Dataset
├── model.h5                   # Trained ANN model
├── scaler.pkl                 # Feature scaler
├── label_encoder_gender.pkl   # Label encoder
├── onehot_encoder_geo.pkl     # One-hot encoder
├── requirements.txt           # Dependencies
├── logs/                      # Training logs
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```
git clone https://github.com/sakshi-0101/customer_churn_ann.git
cd customer_churn_ann
```

### 2. Install dependencies
```
pip install -r requirements.txt
```

### 3. Run the application
```
streamlit run app.py
```

---

## 📊 Dataset Features

The dataset contains the following key features:

- CreditScore  
- Geography  
- Gender  
- Age  
- Tenure  
- Balance  
- Number of Products  
- Has Credit Card  
- Is Active Member  
- Estimated Salary  

---

## 🔍 Workflow

1. Data Preprocessing  
   - Handling categorical variables (Label Encoding & One-Hot Encoding)  
   - Feature scaling using StandardScaler  

2. Model Building  
   - ANN with multiple dense layers  
   - Activation functions: ReLU & Sigmoid  

3. Model Training  
   - Binary Crossentropy Loss  
   - Optimization using Adam  

4. Prediction  
   - Real-time prediction using Streamlit app  

---

## 📈 Features of the App

- Interactive UI using Streamlit  
- Real-time churn prediction  
- Pre-trained model integration  
- User-friendly input system  

---

## 📦 Requirements

```
pandas
numpy
scikit-learn
tensorflow
streamlit
```



## 👩‍💻 Author

**Sakshi Grawal**
