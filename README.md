# Telco Customer Churn Prediction

This project uses the **Teclo-Customer-Churn** dataset to train a machine learning classification model that predicts whether a customer is likely to churn.
Trained model is then used to make new predictions on newly generated customers stored in a database.

This project is a learning exercise on end-to-end ML pipeline.

---

## Project Overview

- Preprocessed the dataset
- trained a model
- created a database and customer generator
- used the trained model on generated customers

---

## Model Performance

- ⚠️ Model performance is still a work in progress.

| Class | Precision | Recall | F1-Score | Support |
|------|----------|--------|---------|---------|
| 0 (No Churn) | 0.91 | 0.77 | 0.83 | 1036 |
| 1 (Churn) | 0.55 | 0.79 | 0.65 | 373 |

**Accuracy:** 0.77  
**Macro Avg F1:** 0.74  
**Weighted Avg F1:** 0.78  

### Confusion Matrix
[[798 238]
 [ 80 293]]

---

## Installation & Setup

- python -m venv venv
- venv\Scripts\activate
- pip install -r requirements.txt
- python database/init_db.py
- python main.py

---

### TODO

- test other models and compare results
- work on accuracy

