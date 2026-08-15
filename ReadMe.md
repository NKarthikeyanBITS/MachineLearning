# Multi-Model ML Classification Dashboard

An interactive Streamlit web application designed to evaluate and run real-time inference on 5 distinct machine learning algorithms.

## 🛠️ Implemented Models
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier
4. Naive Bayes Classifier (Gaussian)
5. Ensemble Model (Random Forest)

## 📊 Evaluation Metrics Calculated
* Accuracy
* AUC Score
* Precision
* Recall
* F1 Score

## 📁 Repository Structure
* `app.py`: Streamlit main dashboard application.
* `requirements.txt`: Python package dependencies.
* `test_data.csv`: Validation dataset used during experimentation.
* `model/`: Directory hosting training pipeline scripts and saved `.pkl` model artifacts.

## 🚀 How to Run Locally
1. Clone repository: `git clone <your-repo-url>`
2. Install dependencies: `pip install -r requirements.txt`
3. Run app: `streamlit run app.py`
