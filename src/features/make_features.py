from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
 
 
def build_preprocessor():
 
    numeric_features = [
        'Age', 'Tenure', 'Usage Frequency',
        'Support Calls', 'Payment Delay',
        'Total Spend', 'Last Interaction'
    ]
 
    categorical_features = [
        'Gender', 'Subscription Type', 'Contract Length'
    ]
 
    numeric_transformer = Pipeline([
        ('scaler', StandardScaler())
    ])
 
    categorical_transformer = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
 
    return ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])