# Medical Insurance Cost Prediction

## Project Overview

This project predicts medical insurance charges based on personal characteristics and lifestyle factors using Machine Learning. A regression model is trained on medical insurance data, saved as a `.pkl` file, and integrated with a FastAPI application to provide predictions through a REST API.

The application can accept user inputs such as age, sex, BMI, number of children, smoking status, and region, then return the estimated insurance charges.

## Objectives

* Understand and preprocess medical insurance data.
* Train a Machine Learning regression model.
* Save and load the trained model using Joblib.
* Build a REST API using FastAPI.
* Validate input data using Pydantic.
* Deploy the application on Render.
* Test predictions using Swagger UI.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib
* FastAPI
* Uvicorn
* Render
* Git and GitHub

## Dataset Description

The dataset contains information about individuals and their medical insurance charges.

| Column     | Description                                 |
| ---------- | ------------------------------------------- |
| `age`      | Age of the individual                       |
| `sex`      | Recorded sex category                       |
| `bmi`      | Body Mass Index                             |
| `children` | Number of children or dependents recorded   |
| `smoker`   | Smoking status (`yes` or `no`)              |
| `region`   | Geographic region                           |
| `charges`  | Medical insurance charges (target variable) |

### Input Features

* Age
* Sex
* BMI
* Children
* Smoker
* Region

### Target Variable

* `charges`

This is a supervised Machine Learning regression problem because the target variable is a continuous numerical value.

## Project Structure

```text
insurance-project/
├── main.py
├── train.py
├── requirements.txt
├── README.md
├── data/
│   └── insurance_cleaned.csv
└── model/
    └── insurance_model.pkl
```

**Note:** The exact filenames and folders should match your actual project. The model file is generated during training.

## Installation and Setup

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_PROJECT_FOLDER>
```

Replace the placeholders with your actual GitHub repository URL and folder name.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
python -m pip install -r requirements.txt
```

## Model Training

Run the training script:

```bash
python train.py
```

The training process should:

1. Load the cleaned dataset.
2. Separate input features and the target variable.
3. Split the data into training and testing sets.
4. Train the selected regression model.
5. Evaluate model performance.
6. Save the trained model as `model/insurance_model.pkl`.

The exact algorithm and evaluation metrics depend on the implementation in `train.py`.

## Run the FastAPI Application Locally

Start the API server:

```bash
uvicorn main:app --reload
```

By default, the application runs at:

```text
http://127.0.0.1:8000
```

### Interactive API Documentation

Open the following URL in your browser:

```text
http://127.0.0.1:8000/docs
```

Use Swagger UI to inspect the available endpoints, enter input values, and test predictions.

## API Usage

The API accepts the individual's characteristics and returns an estimated medical insurance charge.

Example input:

```json
{
  "age": 31,
  "sex": "female",
  "bmi": 25.74,
  "children": 0,
  "smoker": "no",
  "region": "southeast"
}
```

The exact request schema must match the Pydantic model defined in `main.py`.

Example response format:

```json
{
  "predicted_charges": 4500.0
}
```

The number above is illustrative, not a verified prediction from the trained model. Actual results depend on your model and preprocessing pipeline.

## Deployment on Render

### 1. Push the project to GitHub

```bash
git add .
git commit -m "Add insurance prediction API"
git push origin main
```

Ensure your repository contains the application code, `requirements.txt`, and the trained model file if the application loads it directly during startup.

### 2. Create a Web Service

1. Sign in to Render.
2. Create a new Web Service.
3. Connect your GitHub repository.
4. Select the appropriate Python runtime.
5. Configure the build and start commands.

Build command:

```bash
pip install -r requirements.txt
```

Start command:

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

6. Deploy the service and review the deployment logs.

### 3. Test the deployed API

After a successful deployment, open:

```text
https://YOUR-RENDER-SERVICE.onrender.com/docs
```

Replace the example hostname with your actual Render service URL.

Use the interactive documentation to test the prediction endpoint.

## Model Evaluation

The regression model can be evaluated using appropriate metrics, such as:

* **MAE (Mean Absolute Error):** Average absolute prediction error.
* **MSE (Mean Squared Error):** Average squared prediction error.
* **RMSE (Root Mean Squared Error):** Error measured on the target variable's scale.
* **R² Score:** Measures how much of the target's variance is explained by the model.

Include the actual scores from your training output when documenting the final model.

## Key Learnings

* Data preprocessing and feature preparation.
* Regression model training and evaluation.
* Model serialization using Joblib.
* Creating REST APIs using FastAPI.
* Input validation using Pydantic.
* Managing Python dependencies.
* Deploying a Machine Learning application on Render.
* Testing APIs through Swagger UI.

## Future Improvements

* Add stronger input validation and error handling.
* Compare multiple regression algorithms.
* Improve model performance through cross-validation and hyperparameter tuning.
* Add logging and automated tests.
* Build a user-friendly frontend.
* Monitor model performance and data drift.

## Disclaimer

This project is intended for educational purposes. Predictions are estimates generated from historical data and should not be treated as official insurance quotations or financial advice.

## Author

**Hari Prakash Allaka**

Machine Learning | Python | SQL | Data Analytics | FastAPI
