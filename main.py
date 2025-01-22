# Install required libraries
# !pip install fastapi uvicorn nest_asyncio pandas scikit-learn joblib

from fastapi import FastAPI, UploadFile, File
import pandas as pd
from pydantic import BaseModel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import nest_asyncio
from fastapi.responses import JSONResponse
import uvicorn

# Initialize FastAPI app
app = FastAPI()

# Global variables to store the dataset and model
data = None
model = None

# Upload Endpoint
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global data
    try:
        # Save the uploaded file to disk
        with open("uploaded_dataset.csv", "wb") as f:
            f.write(file.file.read())
        
        # Load the dataset into memory
        data = pd.read_csv("uploaded_dataset.csv")
        return {"message": "Dataset uploaded successfully", "columns": list(data.columns)}
    except Exception as e:
        return {"error": str(e)}

# Train Endpoint
@app.post("/train")
async def train_model():
    global data, model
    if data is None:
        return {"error": "No dataset uploaded. Please upload a dataset first."}
    try:
        # Extract features and target
        X = data[['Air temperature [K]', 'Tool wear [min]']]  # Update based on dataset
        y = data['Target']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train Logistic Regression model
        model = LogisticRegression()
        model.fit(X_train, y_train)

        # Evaluate the model
        accuracy = accuracy_score(y_test, model.predict(X_test))
        joblib.dump(model, "model.pkl")  # Save the trained model

        return {"message": "Model trained successfully", "accuracy": accuracy}
    except KeyError as e:
        return {"error": f"Missing column: {e}"}
    except Exception as e:
        return {"error": str(e)}  # Catch and return any other errors

# Predict Input Schema
class PredictInput(BaseModel):
    AirTemp: float
    ToolWear: int

# Predict Endpoint
@app.post("/predict")
async def predict(input_data: PredictInput):
    global model
    if model is None:
        return {"error": "No model trained. Please train a model first."}

    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data.dict()])

    # Make prediction
    prediction = model.predict(input_df)[0]
    confidence = max(model.predict_proba(input_df)[0])

    return {
        "Downtime": "Yes" if prediction == 1 else "No",
        "Confidence": round(confidence, 2)
    }

# Run FastAPI in Jupyter Notebook
nest_asyncio.apply()  # Allow FastAPI to run inside Jupyter Notebook
uvicorn.run(app, host="127.0.0.1", port=8000)

