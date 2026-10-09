from pathlib import Path
import numpy as np

import pandas as pd
import joblib
from sklearn import pipeline
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# 1. Define project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "insurance_cleaned.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "insurance_model.pkl"

# 2. Load dataset
df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape) 
print("\nFirst five rows:") 
print(df.head()) 

# 3. Separate features and target 

X = df.drop(columns=["charges"])
y = df["charges"] 
numerical_features = ["age", "bmi", "children"]
categorical_features = ["sex", "smoker", "region"] 

# 4. Create preprocessing step 

preprocessor = ColumnTransformer( 
    transformers=[ 
        ( "categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features, ), ("numerical", "passthrough", numerical_features), ] )

# 5. Create complete ML pipeline 

pipeline = Pipeline( steps=[ ("preprocessor", preprocessor), ("regressor", LinearRegression()), ] ) 

# 6. Split into training and testing data 

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42, ) 

#7. Train the model 

pipeline.fit(X_train, y_train) 
print("\nModel training completed!") 

# Make predictions on test data 

y_pred = pipeline.predict(X_test) 

#9. Evaluate the model 

mae = mean_absolute_error(y_test, y_pred) 
rmse = np.sqrt(mean_squared_error(y_test, y_pred)) 
r2 = r2_score(y_test, y_pred) 
print("\nModel Evaluation") 
print("----------------") 
print(f"MAE: {mae:.2f}") 
print(f"RMSE: {rmse:.2f}") 
print(f"R2: {r2:.4f}") 

# 10. Save the complete pipeline

MODEL_DIR.mkdir(parents=True, exist_ok=True) 
joblib.dump(pipeline, MODEL_PATH) 
print(f"\nModel saved successfully at: {MODEL_PATH}")