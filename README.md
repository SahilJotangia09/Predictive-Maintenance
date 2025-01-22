End-to-End Machine Learning Project using FastAPI

This repository contains an end-to-end machine learning project built with FastAPI. It provides APIs to upload a dataset, train a machine learning model, and make predictions based on the trained model. The project uses Logistic Regression for modeling and supports interaction via Postman or any API testing tool.

Features

Upload Dataset: Upload a CSV dataset via the /upload endpoint.

Train Model: Train a Logistic Regression model on the uploaded dataset via the /train endpoint.

Make Predictions: Use the /predict endpoint to predict outcomes based on the trained model.

Model Persistence: Save and load the trained model using joblib.

Prerequisites

Ensure you have the following installed:

Python 3.8+

Pip

Install the required libraries using the command:

pip install fastapi uvicorn nest_asyncio pandas scikit-learn joblib

Folder Structure

project-directory/
|-- app.py               # Main application file
|-- uploaded_dataset.csv # Dataset (uploaded at runtime)
|-- model.pkl            # Trained model (saved after training)
|-- README.md            # Project documentation

API Endpoints

1. Upload Dataset

Endpoint: POST /upload

Description: Upload a CSV dataset. The dataset is expected to contain the following columns:

Air temperature [K]

Tool wear [min]

Target

Request Example:

curl -X POST "http://127.0.0.1:8000/upload" -F "file=@path/to/your/dataset.csv"

Response Example:

{
  "message": "Dataset uploaded successfully",
  "columns": ["Air temperature [K]", "Tool wear [min]", "Target"]
}

2. Train Model

Endpoint: POST /train

Description: Train a Logistic Regression model using the uploaded dataset.

Request Example:

curl -X POST "http://127.0.0.1:8000/train"

Response Example:

{
  "message": "Model trained successfully",
  "accuracy": 0.85
}

3. Make Predictions

Endpoint: POST /predict

Description: Predict the downtime based on input features.

Input Schema:

Field

Type

Description

AirTemp

float

Air temperature in Kelvin

ToolWear

int

Tool wear in minutes

Request Example:

curl -X POST "http://127.0.0.1:8000/predict" \
-H "Content-Type: application/json" \
-d '{"AirTemp": 300.5, "ToolWear": 120}'

Response Example:

{
  "Downtime": "Yes",
  "Confidence": 0.92
}

How to Run the Application

Clone the repository:

git clone https://github.com/your-repo-name.git
cd your-repo-name

Install the required dependencies:

pip install -r requirements.txt

Start the FastAPI server:

python app.py

Open your browser or Postman and interact with the endpoints at:

http://127.0.0.1:8000

Example Workflow

Upload the Dataset: Use the /upload endpoint to upload your dataset.

Train the Model: Use the /train endpoint to train the Logistic Regression model.

Make Predictions: Use the /predict endpoint to make predictions based on the input data.

Technologies Used

FastAPI: For building the RESTful API.

scikit-learn: For training the Logistic Regression model.

Pandas: For data manipulation and preprocessing.

Joblib: For saving and loading the trained model.

Postman: For testing the API endpoints.
