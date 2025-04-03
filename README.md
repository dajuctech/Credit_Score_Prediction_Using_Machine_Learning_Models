# Credit Score Prediction Using Machine Learning Models

This project aims to predict creditworthiness and analyze financial behavior using both supervised and unsupervised machine learning techniques. The workflow is organized into several key phases:

---

## 1. Problem Definition
- **Supervised Learning:** Predict creditworthiness.
- **Unsupervised Learning:** Analyze customer behavior patterns.

---

## 2. Data Collection & Preparation
- **Library Installation & Import:**  
  Installed necessary packages (e.g., `umap-learn`, `imblearn`, `xgboost`) and imported libraries for data analysis, visualization, and machine learning.
- **Data Loading:**  
  Fetched the dataset from any source.
- **Dataset Reshuffling & Reduction:**  
  Shuffled the data for randomness, reduced the dataset to 20% of its original size for efficiency, and reset the index.

---

## 3. Exploratory Data Analysis (EDA)
- **Dataset Inspection:**  
  Examined the shape, data types, and memory usage of the dataset.
- **Visualizations:**  
  - Bar charts for data type distribution and memory usage.
  - Histograms and boxplots to assess numerical feature distributions and detect outliers.
  - Heatmaps for missing value patterns.
  - Correlation matrix for numerical features.
  - Count plots for target variable distribution.
- **Duplicate Detection:**  
  Identified and visualized duplicate records.

---

## 4. Data Preprocessing
- **Missing Data & Data Type Correction:**  
  Identified missing values, corrected data types, and classified columns as categorical or numerical.
- **Encoding & Cleaning Categorical Variables:**  
  Applied ordinal and one-hot encoding on columns such as Month, Occupation, Credit Mix, Credit Score, Payment Behavior, etc.
- **Handling Numerical Data:**  
  Converted mixed-format entries (e.g., “years and months” to total months), ensured numeric conversion, and filled missing values using median imputation.
- **Irrelevant Columns:**  
  Dropped columns not useful for modeling.
- **Saving Cleaned Data:**  
  Exported the processed dataset to a CSV file.

---

## 5. Outlier Detection and Removal
- **Techniques Applied:**  
  Utilized both the Interquartile Range (IQR) method and Z-score analysis to identify and remove outlier data points.

---

## 6. Data Standardization
- **Scaling:**  
  Standardized features using `StandardScaler` to prepare for dimensionality reduction and improve model performance.

---

## 7. Handling Imbalanced Classes
- **Oversampling:**  
  Applied `RandomOverSampler` to balance the classes in the target variable.

---

## 8. Dimensionality Reduction
- **PCA & UMAP:**  
  Reduced the dataset to 2 dimensions using both PCA and UMAP.
- **Visualization:**  
  Created scatter plots to visualize the reduced data.
- **Classifier Evaluation:**  
  Trained a Decision Tree classifier on both PCA and UMAP embeddings to compare performance.
- **Data Export:**  
  Saved the dimensionality-reduced data to CSV files.

---

## 9. Supervised Model Development
- **Data Splitting:**  
  Divided the PCA-reduced data into training, validation, and test sets.
- **Model Training & Evaluation:**  
  Built and evaluated several classifiers including:
  - Logistic Regression
  - Random Forest
  - K-Nearest Neighbors (KNN)
  - Naive Bayes
  - XGBoost
  - Support Vector Machine (SVM)
  - Decision Tree Classifier
- **Performance Metrics:**  
  Measured accuracy, precision, recall, F1 score, and visualized confusion matrices.
- **Model Comparison:**  
  Compared model performances using bar charts.

---

## 10. Unsupervised Learning: Clustering
- **KMeans Clustering:**  
  - Determined the optimal number of clusters using the Elbow Method.
  - Calculated the silhouette score.
  - Visualized clusters and centroids using scatter plots.
- **Hierarchical Clustering:**  
  - Applied Agglomerative Clustering and plotted dendrograms.
  - Compared clustering metrics between normal and oversampled datasets.

---

## 11. Ensemble Learning Techniques
- **Bagging:**  
  - Implemented using Decision Trees.
  - Evaluated with cross-validation and plotted ROC curves.
- **Boosting (AdaBoost):**  
  - Trained an AdaBoost classifier.
  - Assessed performance with ROC curves.
- **Voting Classifier:**  
  - Combined Logistic Regression, Decision Tree, and SVM.
  - Evaluated using cross-validation and ROC analysis.
- **Ensemble Comparison:**  
  Compared accuracy and macro-average AUC-ROC scores across ensemble methods using bar charts.


