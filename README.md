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
├── tests/  
│    └── conftest.py
│    └── test_features.py
│    └── test_make_dataset.py
│    └── test_evaluate.py
│    └── test_voting.py
│
├── .github/workflows
│    └── ci.yml
│
└── src/                        <- Source code for use in this project
    ├── __init__.py            <- Makes src a Python module
    │
    ├── data/                  <- Scripts to load and process data
    │   └── make_dataset.py      <- Data loading and cleaning functions
    │
    ├── features/             <- Scripts to turn raw data into features for modeling
    │   └── make_features.py <- Feature engineering and transformation
    │
    ├── models/               <- Scripts to train models and make predictions
    │   ├── train.py    <- Model training scripts
    │   └── voting.py  <- Prediction functions
    │   └── evaluate.py <- Metrics and evaluation functions
├── app/                             <- FastAPI application
│   ├── main.py                      <- API routes, middleware, exception handlers
│   ├── auth.py                      <- API key authentication + role-based authorization
│   ├── config.py                    <- Settings loaded from .env via pydantic-settings
│   ├── schemas.py                   <- Request/response models with full validation
│   ├── model_loader.py              <- Startup model cache
│   ├── middleware.py                <- Request logging + X-Request-ID tracing
│   └── ui.py                        <- Streamlit interactive UI
    └── train_pipeline.py     <- End-to-end training and evaluation pipeline
```

## How to Run the Project

1. **Clone Repository**
   ```bash
   git clone <repository-url>
   cd ChurnPredictionProject
   ```

2. **Set Up Virtual Environment**
   
   Create virtual environment:
   ```bash
   python -m venv venv
   ```
   
   Activate it:
   
   *Windows:*
   ```bash
   venv\Scripts\activate
   ```
   
   *macOS/Linux:*
   ```bash
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Set up environment variables**
   # Windows
   ```copy .env.example .env
   ```
   # macOS/Linux
   ```cp .env.example .env
   ```
5. **Run Training Pipeline**
   ```bash
   python -m src.train_pipeline
   ```
   *or*
   ```bash
   python src/train_pipeline.py
   ```
6. **Start the API**
   ```uvicorn app.main:app --reload
   ```
7. **Start the UI**
   ```streamlit run app/ui.py
   ```

Train pipeline will:

- Generate EDA visualizations

- Train all models

- Print evaluation metrics

- Save plots and SHAP analysis

- Output final best model performance

### Output Artifacts

The models/ directory will contain:
- model .pkls
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

### License

This project is for educational and portfolio purposes.
