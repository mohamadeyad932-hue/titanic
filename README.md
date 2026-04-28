# 🚢 Titanic Survival Prediction: Data Preprocessing Pipeline

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

---

## 🌐 Overview
This project demonstrates a robust data preprocessing pipeline for the classic Titanic dataset from Kaggle. The primary goal is to transform raw, "messy" data into a clean, structured format optimized for Machine Learning models. This repository showcases my ability to handle missing values, perform feature engineering, and prepare datasets for predictive analysis.

---

## 🛠️ Tech Stack
* **Python** (Core Logic)
* **Pandas & NumPy** (Data Manipulation & Analysis)
* **Scikit-Learn** (Preprocessing Tools)
* **Seaborn & Matplotlib** (Data Visualization)
* **Kagglehub** (Direct Cloud Data Integration)

---

## 🚀 Key Engineering Steps

### 1. Automated Data Acquisition
* Integrated `kagglehub` to fetch the latest dataset versions directly, ensuring an automated and reproducible workflow without the need for manual downloads.

### 2. Advanced Missing Value Imputation
* **Age:** Imputed missing values using the **Mean** to preserve the statistical distribution of the feature.
* **Cabin:** Dropped the column entirely due to extreme nullity (>70%), preventing the introduction of model noise.
* **Embarked:** Utilized **Mode Imputation** to fill categorical gaps with the most frequent port of embarkation.

### 3. Intelligent Feature Encoding
* **One-Hot Encoding:** Applied to low-cardinality features (`Sex`, `Embarked`) to prevent the model from assuming an incorrect mathematical or ordinal relationship.
* **Label Encoding (Factorization):** Applied to high-cardinality features (`Name`, `Ticket`) to convert complex strings into unique numerical identifiers efficiently.

---

## 📊 Final Result
The output is a fully numerical `df_final` dataframe. It is completely free of null values and structurally optimized for high-performance ML algorithms, such as Decision Trees (which is initialized in the notebook).

---

## 📂 How to Run
1. Clone this repository:
   ```bash
   git clone [Your-Repository-URL]
