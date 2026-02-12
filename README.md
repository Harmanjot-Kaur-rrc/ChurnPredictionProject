# Customer Churn Prediction

Dataset Link: https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset

- This project builds and evaluates multiple machine learning models to predict customer churn. The objective is to identify customers likely to leave a service subscription and provide actionable insights to support retention strategies.

- The project follows a modular structure separating data processing, modeling, evaluation, and visualization to ensure reproducibility and maintainability.


### Project Structure

```
├── README.md                    <- The top-level README for developers using this project
├── requirements.txt             <- The requirements file for reproducing the environment
├── .gitignore                   <- Git ignore file
│
├── data/                        <- All dataset files
│   ├── raw/                    <- Original, immutable dataset files
│   └── processed/              <- Cleaned dataset used for modeling
│
├── notebooks/                   <- Jupyter notebooks for EDA and experimentation
│
├── models/                     <- Saved model files, visualizations, and evaluation outputs
│
└── src/                        <- Source code for use in this project
    ├── __init__.py            <- Makes src a Python module
    │
    ├── data/                  <- Scripts to load and process data
    │   └── load_data.py      <- Data loading and cleaning functions
    │
    ├── features/             <- Scripts to turn raw data into features for modeling
    │   └── build_features.py <- Feature engineering and transformation
    │
    ├── models/               <- Scripts to train models and make predictions
    │   ├── train_model.py    <- Model training scripts
    │   └── predict_model.py  <- Prediction functions
    │
    ├── evaluation/           <- Scripts to evaluate model performance
    │   └── evaluate_model.py <- Metrics and evaluation functions
    │
    ├── visualization/        <- Scripts to create visualizations
    │   └── visualize.py     <- Plotting and visualization functions
    │
    └── train_pipeline.py     <- End-to-end training and evaluation pipeline
```

### How to Run the Project
1. **Clone Repository**
  git clone <your-repo-url>
  cd ChurnPredictionProject

2. **Create Virtual Environment**
  python -m venv venv
  venv\Scripts\activate     # Windows

4. **Install Dependencies**
  pip install -r requirements.txt
  
5. **Run Training Pipeline**
  python -m src.train_pipeline


This will:

- Generate EDA visualizations

- Train all models

- Print evaluation metrics

- Save plots and SHAP analysis

- Output final best model performance

### Output Artifacts

The models/ directory will contain:

- churn_distribution.png
- numerical_distributions.png
- categorical_distributions.png
- correlation_heatmap.png
- confusion_matrix.png
- roc_curve.png
- feature_importance.png
- shap_summary.png

### Reproducibility

- All dependencies listed in requirements.txt

- No hard-coded paths outside project structure

- Pipeline executable via module (python -m)

### Future Improvements

- CI/CD integration

- Deployment-ready API endpoint

### License

This project is for educational and portfolio purposes.
