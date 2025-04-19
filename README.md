🚨 **Fraudulent Claim Detection**
A machine learning project focused on identifying fraudulent insurance claims using Python and data science techniques.

📂 **Project Overview**
Insurance fraud can lead to billions in losses annually. This project explores a real-world dataset to build a machine learning model that classifies insurance claims as fraudulent or legitimate.

🔧 **Tools & Technologies**
1. Python
2. Pandas, NumPy
3. Matplotlib, Seaborn
4. Scikit-learn
5. Jupyter Notebook

📊 **Key Steps**
1. Data Loading & Exploration
  - Understand the structure of the dataset.
  - Explore class imbalance and claim distributions - RandomOverSampler technique.

2. Data Preprocessing
  - Handle missing values
  - Encode categorical variables
  - Normalize/scale features
  - Binning

3. Model Building
   1. **Logistic Regression Model**
       - Feature Selection using RFECV – Identify the most relevant features using Recursive Feature Elimination with Cross-Validation.
       - Model Building and Multicollinearity Assessment – Build the logistic regression model and analyse statistical aspects such as p-values and VIFs to detect multicollinearity.
       - Model Training and Evaluation on Training Data – Fit the model on the training data and assess initial performance.
       - Finding the Optimal Cutoff – Determine the best probability threshold by analysing the sensitivity-specificity tradeoff and precision-recall tradeoff.
       - FInal Prediction and Evaluation on Training Data using the Optimal Cutoff – Generate final predictions using the selected cutoff and evaluate model performance.
   2. **Random Forest Model**
       - Get Feature Importances - Obtain the importance scores for each feature and select the important features to train the model.
       - Model Evaluation on Training Data – Assess performance metrics on the training data.
       - Check Model Overfitting using Cross-Validation – Evaluate generalisation by performing cross-validation.
       - Hyperparameter Tuning using Grid Search – Optimise model performance by fine-tuning hyperparameters.
       - Final Model and Evaluation on Training Data – Train the final model using the best parameters and assess its performance.

4. Evaluation
  - Confusion matrix, accuracy, precision, recall, F1-score
  - ROC-AUC curve

✅ **Results**
Best performing model: Random Forest

Achieved an accuracy of 80% and precision of 60% on the test set.

