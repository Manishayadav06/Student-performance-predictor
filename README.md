Student Performance Predictor

An AI/ML-based regression system that predicts student exam scores using Machine Learning algorithms and educational performance factors.

Overview

The Student Performance Predictor is a Machine Learning project developed using Python and Scikit-learn. The system analyzes multiple academic and lifestyle factors such as attendance, study hours, parental involvement, motivation level, and previous scores to predict a student's exam performance.

The project compares multiple regression algorithms and evaluates their performance using standard machine learning metrics.

Features Predicts student exam scores using ML models Compares multiple regression algorithms Performs data preprocessing and encoding Generates professional visualizations and reports Displays feature importance analysis Evaluates model accuracy using multiple metrics Supports single-student score prediction Technologies Used Programming Language Python Libraries & Frameworks Pandas NumPy Scikit-learn Matplotlib Seaborn Machine Learning Models Used Linear Regression Ridge Regression Random Forest Regressor Gradient Boosting Regressor Evaluation Metrics

The models are evaluated using:

R² Score Mean Absolute Error (MAE) Root Mean Squared Error (RMSE) Cross Validation Score Dataset Information

Dataset: StudentPerformanceFactors.csv

The dataset contains:

6,607 student records 19 input features Target column: Exam_Score Input Features

Some important features used for prediction include:

Hours Studied Attendance Previous Scores Motivation Level Sleep Hours Teacher Quality Internet Access Family Income Physical Activity Parental Involvement Project Structure Student-Performance-Predictor/ │ ├── student_performance_predictor.py ├── StudentPerformanceFactors.csv ├── student_performance_report.png ├── requirements.txt └── README.md Workflow Load Dataset ↓ Data Preprocessing ↓ Label Encoding ↓ Train-Test Split ↓ Model Training ↓ Performance Evaluation ↓ Visualization & Prediction Data Preprocessing

The following preprocessing steps are applied:

Missing value removal Label Encoding for categorical data Feature selection Train-test splitting Visualizations Generated

The project automatically generates:

Exam Score Distribution Actual vs Predicted Scores Residual Distribution Feature Importance Chart Model Comparison Graph Correlation Heatmap Attendance vs Exam Score Study Hours vs Exam Score Installation Clone the Repository git clone https://github.com/Manishayadav06/Student-performance-predictor Navigate to Project Folder cd Student-Performance-Predictor Install Dependencies pip install -r requirements.txt Requirements

Create a requirements.txt file with:

pandas numpy matplotlib seaborn scikit-learn Run the Project python student_performance_predictor.py Sample Prediction

Example prediction using:

Attendance: 92% Study Hours: 25 Previous Score: 85

The model predicts the expected exam score for the student.

Output

The system:

Prints model evaluation results Selects the best-performing model Generates visualization reports Predicts student performance Future Enhancements Streamlit Web Application Real-time student dashboard Hyperparameter tuning using GridSearchCV Deep Learning implementation Database integration Student performance recommendation system Learning Outcomes

This project helped in understanding:

Machine Learning workflow Regression algorithms Data preprocessing Feature engineering Model evaluation Data visualization Predictive analytics Author

Rainaboina Manisha

Computer Science Engineering (DS)

GitHub: Manishayadav06 GitHub

LinkedIn: Manisha yadav LinkedIn
