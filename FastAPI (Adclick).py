# 1. Library imports
import uvicorn
from fastapi import FastAPI, HTTPException
import numpy as np
import joblib
import pandas as pd
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Class which describes Ad Click measurements
class AdClick(BaseModel):
    UserID: int
    Gender_Male: bool  # Assuming Male is one category
    Gender_Female: bool  # Assuming Female is another category
    Age: int
    EstimatedSalary: int

# 2. Create the app object
app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can specify allowed origins instead of '*' for better security
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Load the model and scaler
full_pipeline = joblib.load("random_forest_optimized_adclick.pkl")  # The model without scaler
scaler = joblib.load("Scaler_adclick.pkl")  # Loading the Scaler

# 3. Predict function
@app.post('/predict/')
def predict_adclick(input_data: AdClick):
    try:
        # Keep UserID for tracking, but exclude it from feature processing
        user_id = input_data.UserID

        # Prepare input values without UserID
        x_values = pd.DataFrame({
            'Gender_Male': [int(input_data.Gender_Male)],
            'Gender_Female': [int(input_data.Gender_Female)],
            'Age': [input_data.Age],
            'EstimatedSalary': [input_data.EstimatedSalary]
        })

        # Apply the scaler to the input data (excluding UserID)
        scaled_input = scaler.transform(x_values)

        # Predict using the model
        prediction = full_pipeline.predict(scaled_input)

        # Convert prediction to string
        prediction_result = 'Have Purchased' if prediction[0] == 1 else 'Have Not Purchased'
        print(scaled_input)
        # Return prediction with the UserID for tracking
        return {'UserID': user_id, 'prediction': prediction_result}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# 5. Run the API with uvicorn (for local testing)
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
