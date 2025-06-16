# AmesHousing
This project demonstrates a full data science workflow using the Ames Housing Dataset, aiming to predict house prices through regression modeling. The project covers data preprocessing, exploratory data analysis, feature engineering, model training, hyperparameter tuning, and evaluation.

## 📂 ames-housing-project/
<pre><strong>├── data/ </strong>
│   └── AmesHousing.csv
<strong>├── notebooks/</strong>
│   └── 01-data-cleaning-and-eda.ipynb
│   └── 02-modeling-and-tuning.ipynb
<strong>├── images/ </strong>
│   └── learning_curve.png
│   └── residuals_plot.png
<strong>├── models/ </strong>
│   └── final_model.pkl
<strong>├── README.md </strong></pre>

## 📊Dataset Overview
Source: Ames Housing Dataset (Kaggle)

Objective: Predict SalePrice using numeric and categorical features.

Target: SalePrice

Number of features (after selection & engineering): 6–8

## 🔧 Project Steps
### 1. Data Cleaning & Feature Selection
Dropped irrelevant or sparse columns:
* Order, PID, Alley, Fence, Mas Vnr Type, Pool QC, Misc Feature

Handled missing values:
* Categorical features: filled with mode or "None" (e.g., Fireplace Qu)
* Numerical features: filled with mean

Selected key numerical features:
* OverallQual, GrLivArea, GarageCars, TotalBsmtSF

Selected categorical features:
* HeatingQC, FireplaceQu

## 2. Feature Engineering
Added new features to improve model accuracy:
* House_Age = YrSold - YearBuilt
* Basement_Ratio = TotalBsmtSF / GrLivArea

These features helped the model capture architectural and spatial characteristics more effectively.

## 3. Preprocessing & Pipeline
Constructed a pipeline with:
* ColumnTransformer for numerical and categorical data
* SimpleImputer for missing values
* StandardScaler for numeric values
* OneHotEncoder for categoricals

Model: RandomForestRegressor

  model = Pipeline([
    ('prep', preprocessor),
    ('rf', RandomForestRegressor(random_state=42))
])

## 📈 Model Performance
### ⚙️ Baseline Model
| Metric | Value    |
| ------ | -------- |
| MSE    | \~1.14e9 |
| R²     | 0.857    |
### 📊 After Feature Engineering
| Metric | Value    |
| ------ | -------- |
| MSE    | \~9.89e8 |
| R²     | 0.877    |
## 🔍 After GridSearchCV Optimization
### Best Parameters:
<pre> {
  "rf__max_depth": 10,
  "rf__min_samples_leaf": 1,
  "rf__min_samples_split": 2,
  "rf__n_estimators": 400
} </pre>
The optimized model showed a slight improvement in performance:

<pre>MSE: 987,349,395
R² Score: 0.8768 </pre>
Although the increase is modest, it reflects a more generalized model, reducing potential overfitting observed in earlier steps.

## 📈 Learning Curve Insights
We also plotted learning curves to visualize model performance across training set sizes. The gap between training and validation scores narrowed significantly, showing reduced overfitting and improved generalization. This provided additional validation of model robustness.

## 🧠 Lessons Learned
Imputation matters: Handling missing data thoughtfully (e.g., treating FireplaceQu as a feature) improved results.
* Feature selection is key: Limiting to the most correlated features avoided overfitting.
* Pipeline usage: Helped maintain a clean and scalable ML workflow.
* Hyperparameter tuning: GridSearchCV significantly streamlined model optimization.
