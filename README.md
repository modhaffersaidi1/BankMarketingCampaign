### Bank Marketing Campaign Prediction
This project aims to compare the performance of various classifiers (k-nearest neighbors, logistic regression, decision trees, and support vector machines) to predict client subscription to a term deposit using a dataset related to the marketing of bank products over the telephone.

### Dataset
The dataset involves direct marketing campaigns of a Portuguese banking institution. The goal is to predict whether a client would subscribe to a term deposit.

### Classifiers and Evaluation Metrics
The classifiers evaluated in this project include:

k-Nearest Neighbors (kNN)

Logistic Regression

Decision Tree

Support Vector Machine (SVM)

Results Summary
- ** KNeighborsClassifier **:
Cross-validation scores: [0.97203791, 0.97203791, 0.9720335, 0.97171749, 0.97155949]

Accuracy: 96.90%

Precision: 0.0

Recall: 0.0

F1-Score: 0.0

- ** LogisticRegression **:
Cross-validation scores: [1.0, 1.0, 1.0, 1.0, 1.0]

Accuracy: 100%

Precision: 100%

Recall: 100%

F1-Score: 100%

- ** DecisionTreeClassifier **:
Cross-validation scores: [1.0, 1.0, 1.0, 1.0, 1.0]

Accuracy: 100%

Precision: 100%

Recall: 100%

F1-Score: 100%

- ** SVC **:
Cross-validation scores: [0.97203791, 0.97203791, 0.9720335, 0.9720335, 0.9720335]

Accuracy: 96.92%

Precision: 0.0

Recall: 0.0

F1-Score: 0.0

### Findings and Recommendations
Findings:
Logistic Regression and Decision Trees performed exceptionally well, achieving perfect accuracy, precision, recall, and F1-scores.

k-Nearest Neighbors (kNN) and Support Vector Machine (SVM) showed high accuracy but failed in precision, recall, and F1-score, indicating they were not effective in predicting the positive class (subscribing to a term deposit).

### Recommendations:
Logistic Regression and Decision Trees are recommended for predicting client subscriptions due to their superior performance.

For future work, consider tuning hyperparameters to further enhance model performance.

Explore additional features or data sources that might improve the predictive power of the models.

Implement the best-performing model in a real-world marketing campaign and monitor its effectiveness.

### How to Use
Clone the repository.

Ensure all necessary libraries are installed (e.g., pandas, scikit-learn, seaborn).

Load the dataset and run the Jupyter Notebook to replicate the analysis and results.

Review the findings and recommendations for actionable insights.

### Conclusion
This project demonstrates the potential of machine learning classifiers in predicting client subscriptions to term deposits. By leveraging accurate models, banks can enhance their marketing strategies and achieve better outcomes in their campaigns.