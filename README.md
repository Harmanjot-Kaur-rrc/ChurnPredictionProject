# PROJECT:  **Customer Churn Prediction**

🎯 Project Overview
This document outlines the comprehensive EDA and modeling approach for predicting customer churn. The goal was to develop a highly accurate model that identifies at-risk customers while sharing data analysis insights to develop retention strategies.

## 1.0 About Auther
Project: Customer Churn Prediction
Auther: Harmanjot Kaur
Date: 10 Feb 2026

## 2.0 About Data
Data: Customer Churn Dataset
Dataset: customer_churn_dataset-training-master.csv  (Kaggle)
         customer_churn_dataset-testing-master.csv


### Column Name	Description
CustomerID:	Unique identifier for each customer
Age:	Age of the customer
Gender:	Gender of the customer (e.g., Male/Female)
Tenure:	How long the customer has been with the company (months/years)
Usage Frequency:	How frequently the customer uses the service
Support Calls:	Number of support/service calls made by the customer
Payment Delay:	Delay in payment
Subscription Type:	Type of subscription (e.g., Basic, Standard, Premium)
Contract Length:	Length of the contract (e.g., monthly, quaterly, yearly)
Total Spend:	Total amount of money spent by the customer
Last Interaction:	Time since the last customer interaction
Churn Target variable:  Whether the customer left (1) or stayed (0)

## ** Exploratory Data Analysis (EDA)**

📁 Dataset Overview
- Total records (train + test): 505,207
- Features: 12 columns (9 numerical, 3 categorical)
- Missing values: 1 row with nulls (removed)
- Duplicates: None found 
- Target variable: Churn (1 = Churned, 0 = Not Churned)


🧹 Data Cleaning Steps

1	Load train & test data:	440,833 + 64,374 rows
2	Concatenate into single DataFrame:	505,207 rows × 12 columns
3	Check missing values:	1 row with nulls in all columns
4	Drop CustomerID (non-predictive)	
5	Drop null row	Clean dataset: 505,206 rows
6	Convert Churn to int:	Target ready for binary classification

🎯 Target Variable: Churn Distribution

- Churn rate: 55.5% (280,492 customers)
- Non-churn: 44.5% (224,714 customers)

📈 Feature Types Summary
	
Numerical: Age, Tenure, Usage Frequency, Support Calls, Payment Delay, Total Spend, Last Interaction, Churn
Categorical: Gender, Subscription Type, Contract Length

📉 Distribution of Numerical Features

- Age, Tenure, Usage Frequency → roughly uniform distributions.
- Support Calls & Payment Delay → right-skewed.
- Total Spend → bimodal distribution.

🔍 Key Insights & Findings

📈 Feature Correlations
Top churn drivers identified:

1. Support Calls (+0.52) - Strongest predictor
2. Payment Delay (+0.33) - Moderate predictor
3. Total Spend (-0.37) - Higher spenders less likely to churn
4. Age (+0.19) - Older customers slightly more likely to churn

📅 Contract Type Analysis

- Monthly: ~90% churn rate (highest risk)
- Quarterly: ~46% churn rate
- Annual: ~46% churn rate
- Finding: Monthly subscribers are most volatile

📊 Demographic Patterns

- Gender distribution: Relatively balanced
- Age range: 18-65 years (uniform distribution)
- Tenure: 1-60 months (newer customers at higher risk)

📞 Support Impact Analysis

- Churned customers show higher median support calls. Support call distribution is right-skewed for churners
- Implication: Customer service issues strongly correlate with attrition

💡 Retention Opportunities

- High-spending customers are loyal - focus on premium retention
- Annual subscribers are stable - encourage long-term commitments
- Early intervention for new customers showing support needs

## **Modeling**

📈 Model Development Workflow
1. Data Strategy
- Split Ratio: 70% Training | 15% Validation | 15% Testing
- Key Consideration: Stratified sampling to preserve churn distribution across all splits

2. Feature Engineering
Preprocessing Pipeline:

- Numeric Features: Standardized using StandardScaler for equal contribution
- Categorical Features: One-hot encoded to avoid ordinal bias
- Pipeline Architecture: Integrated preprocessing with modeling for consistency

🏗️ Model Selection & Justification
# Baseline Models
1. Logistic Regression

- Purpose: Establish interpretable baseline
- Strength: Linear relationships, probabilistic outputs
- Limitation: Assumes linearity between features and log-odds

2. Random Forest

- Purpose: Capture non-linear patterns and interactions
- Strength: Robust to outliers, feature importance metrics
- Optimization: Tuned tree depth and split criteria

3. Gradient Boosting

- Purpose: Sequential error correction
- Strength: High predictive accuracy, handles imbalanced data well

## Advanced Models
1. XGBoost
- Primary Choice: Selected as final model
- Why It was selected: Built-in regularization, efficient computation, excellent performance
- Advantage: Native handling of missing values and feature importance

2. Neural Network (MLP)

- Purpose: Test deep learning capability
- Outcome: Competitive but computationally intensive

3. Voting Ensemble

- Strategy: Combined multiple models with weighted voting
- Insight: XGBoost received double weight due to superior performance
- Result: Slightly improved robustness but marginal gain

📊 Performance Evaluation
## Validation Set Results

**Model: (ROC-AUC, Accuracy, Precision, Recall, F1-Score)**	
**Gradient Boosting**:	(0.9535,	0.9317,	0.8973,	0.9902,	0.9415)	
**XGBoost**:	(0.9530,	0.9316,	0.8978,	0.9894,	0.9414)	
**Random Forest**:	(0.9528,	0.9352,	0.8974,	0.9972,	0.9447)	
**MLP (Neural Network)**:	(0.9527,	0.9350,	0.8975,	0.9967,	0.9445)	
**Voting Ensemble**:	(0.9530,	0.9321,	0.8978,	0.9905,	0.9419)	
**Logistic Regression**:	(0.9091,	0.8480,	0.8734,	0.8494)

Optimized Threshold: 0.148 (maximizes F1-score)
Business Impact: Achieves 99.9% recall while maintaining 89.7% precision
False Negatives: Only 46 churners missed in validation

🔍 Feature Importance Insights
- Business Implications
- Focus Areas: Improve support experience and payment processes
- Retention Strategy: Target monthly contract customers
- Resource Allocation: Prioritize high-risk segments identified by model

## 🏆 Final Model Selection: XGBoost
Why XGBoost?
1. Performance: Virtually tied with best model (0.9530 ROC-AUC)
2. Efficiency: Faster predictions than ensemble methods
3. Interpretability: Clear feature importance rankings
4. Robustness: Handles outliers and missing data effectively
5. Production Ready: Low memory footprint, fast inference

## Final Test Performance
- ROC-AUC: 0.9541 (Excellent discrimination ability)
- Precision-Recall AUC: 0.9545 (Strong for imbalanced data)
- Recall: 99.9% (Minimal missed churners)
- F1-Score: 0.9453 (Balanced performance metric)

✅ Success Criteria Met
- High Accuracy: Exceeded 95% ROC-AUC target
- Business Alignment: Optimized for high recall (don't miss churners)
- Interpretability: Clear feature importance for stakeholder buy-in
- Scalability: Model efficient for production deployment
- Robustness: Consistent performance across data splits




 
