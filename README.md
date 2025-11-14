🩺 Diabetes Prediction using Machine Learning

Logistic Regression | KNN | SVM | Random Forest

This project predicts whether a patient is Diabetic (1) or Not Diabetic (0) using the Pima Indians Diabetes Dataset.
It covers the complete ML pipeline — from data preprocessing to prediction on new samples.

📁 Dataset

The dataset contains the following medical features:

Pregnancies

Glucose

BloodPressure

SkinThickness

Insulin

BMI

DiabetesPedigreeFunction

Age

Outcome (Target: 0 = Not Diabetic, 1 = Diabetic)

🚀 Workflow / Project Phases
Phase 1 — Data Collection

Load dataset from Google Drive

View dataset structure: head(), info(), describe()

Phase 2 — Data Preprocessing

✔ Replace invalid zero values with NaN
✔ Fill missing values using median
✔ Split into features (X) and labels (y)
✔ Apply StandardScaler for normalization

-- Preprocessed columns:
'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI'

Phase 3 — Exploratory Data Analysis (EDA)

Performed visual and statistical analysis:

Distribution of all features (Histogram + KDE)

Correlation heatmap

Outcome class distribution (Imbalanced dataset)

Libraries used:

Matplotlib

Seaborn

Phase 4 — Model Selection & Training

Four ML models were trained:

🔹 Logistic Regression

Class-balanced training

Good baseline performance

🔹 K-Nearest Neighbors (KNN)

Works well with scaled data

Non-linear model

🔹 Random Forest Classifier

Handles missing & noisy data

Strong performance

🔹 Support Vector Machine (SVM)

Linear Kernel

Effective on high-dimensional data

Models were evaluated using:

Accuracy Score

Classification Report

Confusion Matrix

Phase 5 — Model Evaluation

Metrics used:

Precision

Recall

F1-score

Support

Confusion Matrix Display

All models were compared to analyze performance differences.

Phase 6 — Prediction on New Samples

Example input:

# [Pregnancies, Glucose, BP, SkinThickness, Insulin, BMI, DPF, Age]
sample1 = [[1, 89, 66, 23, 94, 28.1, 0.167, 21]]  
sample2 = [[5, 150, 85, 30, 200, 35.0, 0.7, 45]]


After scaling:

Sample 1 → Not Diabetic  
Sample 2 → Diabetic  

🛠️ Technologies Used

Python

Pandas, NumPy

Scikit-Learn

Matplotlib, Seaborn

Google Colab / Jupyter Notebook

📦 Installation
pip install numpy pandas matplotlib seaborn scikit-learn

▶️ How to Run the Project

Clone the repository

Install dependencies

Open the notebook / Python file

Run all cells step-by-step

📈 Future Enhancements

Hyperparameter Tuning (RandomizedSearchCV / GridSearchCV)

Apply SMOTE for handling class imbalance

Save model using Pickle

Build a Streamlit web app for real-time predictions

Deploy to cloud (Render / HuggingFace / AWS)
