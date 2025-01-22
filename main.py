import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import uvicorn

# Initialize FastAPI app
app = FastAPI()

# Global variables
data = None
model = None

# Define input schema for prediction
class PredictInput(BaseModel):
    Temperature: float
    Run_Time: int

# Endpoint: Upload dataset
@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    global data
    if file.filename.endswith('.csv'):
        data = pd.read_csv("predictive_maintenance.csv")
        if {'Machine_ID', 'Temperature', 'Run_Time', 'Downtime_Flag'}.issubset(data.columns):
            return {"message": "Dataset uploaded successfully!"}
        else:
            data = None
            return {"error": "Dataset does not contain required columns."}
    return {"error": "Please upload a valid CSV file."}

# Endpoint: Train model
@app.post("/train")
async def train_model():
    global data, model
    if data is None:
        return {"error": "No dataset uploaded. Please upload a dataset first."}

    # Prepare data
    X = data[["Temperature", "Run_Time"]]
    y = data["Downtime_Flag"].map({"Yes": 1, "No": 0})  # Convert labels to binary
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return {"message": "Model trained successfully!", "accuracy": accuracy, "f1_score": f1}

# Endpoint: Predict downtime
@app.post("/predict")
async def predict_downtime(input_data: PredictInput):
    global model
    if model is None:
        return {"error": "Model not trained. Please train the model first."}

    # Make prediction
    input_df = pd.DataFrame([input_data.dict()])
    prediction = model.predict(input_df)[0]
    confidence = max(model.predict_proba(input_df)[0])

    return {"Downtime": "Yes" if prediction == 1 else "No", "Confidence": round(confidence, 2)}

# Run the app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
