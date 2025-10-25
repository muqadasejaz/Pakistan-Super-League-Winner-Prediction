# 🏏 Pakistan Super League (PSL) — Match Winner Prediction

A machine learning project that predicts the winner of Pakistan Super League (PSL) matches using historical data. It includes data preprocessing, exploratory analysis, and model training using Random Forest , achieving high accuracy in predicting match outcomes.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 📘 Project Overview

This project aims to predict the **winner of Pakistan Super League (PSL)** cricket matches using historical match data.  
The model analyzes factors such as the playing teams, toss results, venues, and other match attributes to estimate the most likely winner.  

This end-to-end solution includes:
- Data preprocessing and exploratory data analysis (EDA)
- Feature encoding and transformation
- Model training and evaluation using top-performing ML algorithm
- Visualization of performance metrics
- Model persistence for deployment and future use

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 🎯 Objective

The main goal of this project is to:
Build a supervised machine learning model that can **predict the winner of PSL matches** based on available historical data.

Specifically, the project focuses on:
- Understanding the relationship between match features (teams, toss, venue, etc.) and outcomes.
- Training multiple ML model
- Evaluating model performance using metrics like accuracy, classification report, and confusion matrix.
- Saving trained pipelines for real-time or batch predictions.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 🌟 Key Features

✅ **Automated dataset discovery** – Automatically locates and loads the PSL dataset from Kaggle environment.  
✅ **Robust preprocessing** – Handles missing data, encodes categorical features, and scales numeric attributes.  
✅ **model trained** –  **Random Forest Classifier**  
✅ **Comprehensive Evaluation** – Accuracy, classification report, and confusion matrix visualization.  
✅ **Label Encoding** – Consistent numeric encoding of team names for compatibility with XGBoost.  
✅ **Prediction on Unseen Data** – Make predictions for new/unseen match records.  
✅ **Model Persistence** – Pipelines and encoders saved using `joblib` for reuse or deployment.  

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 🧰 Tools & Technologies

| Category | Tools/Packages |
|-----------|----------------|
| Language | Python 3.x |
| Libraries | `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `joblib` |
| Environment | Kaggle / Jupyter Notebook |
| Visualization | `matplotlib`, `ConfusionMatrixDisplay` |
| Model Saving | `joblib` |

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 📊 Dataset

**Dataset Source:** [Pakistan Super League Dataset on Kaggle](https://www.kaggle.com/datasets/brandmustafa/pakistan-super-league/data)

**Dataset Description:**  
The dataset includes detailed match-level information from PSL seasons, such as:
- `team1`, `team2` — competing teams  
- `venue`, `city` — match location details  
- `toss_winner`, `toss_decision` — toss results  
- `result`, `winner` — final match outcome  

**Target Variable:**  
`winner` — The team that won the match.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## ⚙️ Workflow

1. **Import libraries and setup environment**
2. **Auto-detect and load dataset from `/kaggle/input`**
3. **EDA** – Inspect missing values, value distributions, and target class balance
4. **Feature Engineering** – Encode categorical variables and scale numerical features
5. **Train-Test Split** (80-20)
6. **Train two models** – RandomForest 
7. **Evaluate models** – Accuracy, classification report, and confusion matrix
8. **Predict on unseen data**
9. **Save trained models and label encoder for future use**

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 🧠 Model Training & Evaluation

### Model Used:
- **RandomForestClassifier**

Both models were trained on preprocessed features using a unified scikit-learn `Pipeline`.  

### Evaluation Metrics:
- Accuracy
- Precision, Recall, F1-Score
- Confusion Matrix Visualization

  <img width="636" height="558" alt="image" src="https://github.com/user-attachments/assets/ebd7c443-1888-4d3c-88a1-5946cc435683" />

  <img width="615" height="292" alt="image" src="https://github.com/user-attachments/assets/33ca7ea7-54f9-49e0-8374-b2d2676bd452" />
  

Example Evaluation Output:

<img width="604" height="221" alt="image" src="https://github.com/user-attachments/assets/42207033-8bfa-490a-a1da-4fbd5116185d" />

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 📚 References

- [Scikit-Learn Documentation](https://scikit-learn.org/stable/)
- [Kaggle PSL Dataset](https://www.kaggle.com/datasets/brandmustafa/pakistan-super-league/data)

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 👤 Author

Muqadas Ejaz

BS Computer Science (AI Specialization)

AI/ML Engineer

Data Science & Gen AI Enthusiast

📫 Connect with me on [LinkedIn](https://www.linkedin.com/in/muqadasejaz/)  

🌐 GitHub: [github.com/muqadasejaz](https://github.com/muqadasejaz)

📬 Kaggle: [Kaggle Profile](https://www.kaggle.com/muqaddasejaz) 

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 📎 License

This project is open-source and available under the [MIT License](LICENSE).
