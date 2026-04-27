import gradio as gr
import pandas as pd
import joblib

# Load trained model
model = joblib.load("churn_prediction.pkl")

# all feature names 
feature_columns = model.feature_names_in_

# Prediction function
def predict_churn(*inputs):
    input_data = pd.DataFrame([inputs], columns=feature_columns)
    prediction = model.predict(input_data)[0]
    return "Churn" if prediction == 1 else "Not Churn"


# UI 
interface = gr.Interface(
    fn=predict_churn,
    inputs=[
        gr.Radio(["Male", "Female",],label="Select Male or Female"),
        gr.Radio([0, 1], label="Select whether the customer is a senior citizen or not (1, 0)"),
        gr.Radio(["Yes", "No",],label="Whether the customer has a partner or not (Yes, No)"),
        gr.Radio(["Yes", "No",],label="Whether the customer has dependents or not (Yes, No)"),
        gr.Number(label="Number of months the customer has stayed with the company"),
        gr.Radio(["Yes", "No",],label="Whether the customer has a phone service or not (Yes, No))"),
        gr.Radio(["Yes", "No","No phone service",],label="Whether the customer has multiple lines or not (Yes, No, No phone service)"),
        gr.Radio(["DSL", "No","Fiber optic"],label="Customer’s internet service provider (DSL, Fiber optic, No)"),
        gr.Radio(["Yes", "No","No internet service"],label="Whether the customer has online security or not (Yes, No, No internet service)"),
        gr.Radio(["Yes", "No","No internet service"],label="Whether the customer has online backup or not (Yes, No, No internet service)"),
        gr.Radio(["Yes", "No","No internet service"],label="Whether the customer has device protection or not (Yes, No, No internet service)"),
        gr.Radio(["Yes", "No","No internet service"],label="Whether the customer has tech support or not (Yes, No, No internet service)"),
        gr.Radio(["Yes", "No","No internet service"],label="Whether the customer has streaming TV or not (Yes, No, No internet service)"),
        gr.Radio(["Yes", "No","No internet service"],label="Whether the customer has streaming movies or not (Yes, No, No internet service)"),
        gr.Radio(["Month-to-month", " One year","Two year"],label="The contract term of the customer (Month-to-month, One year, Two year)"),
        gr.Radio(["Yes", "No",],label="Whether the customer has paperless billing or not (Yes, No)"),
        gr.Radio(["Electronic check", "Mailed check","Bank transfer (automatic)"," Credit card (automatic)"],label="The customer’s payment method (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic)"),
        gr.Number(label="The amount charged to the customer monthly"),
        gr.Number(label="The total amount charged to the customer"),

    ],
    
    outputs="text",
    title="Customer Churn Prediction App",
    description="Enter customer data to predict whether they will churn or not."
)

interface.launch()