import streamlit as st
import pandas as pd
import tensorflow as tf

# Function to convert checkbox values to 1 and 0
def checkbox_to_int(value):
    return 1 if value else 0

# Function to get user input for each column
def get_user_input():
    st.sidebar.title("User Input")
    credit_score = st.sidebar.text_input("Credit Score", "650")
    age = st.sidebar.text_input("Age", "30")
    tenure = st.sidebar.text_input("Tenure", "5")
    balance = st.sidebar.text_input("Balance", "50000")
    num_of_products = st.sidebar.text_input("Num of Products", "2")
    has_cr_card = checkbox_to_int(st.sidebar.checkbox("Has Credit Card", value=True))
    is_active_member = checkbox_to_int(st.sidebar.checkbox("Is Active Member", value=True))
    estimated_salary = st.sidebar.text_input("Estimated Salary", "100000")
    is_germany = checkbox_to_int(st.sidebar.checkbox("Is from Germany", value=False))
    is_spain = checkbox_to_int(st.sidebar.checkbox("Is from Spain", value=False))
    is_male = 1 if st.sidebar.radio("Gender", ["Male", "Female"]) == "Male" else 0

    return {
        "CreditScore": int(credit_score),
        "Age": int(age),
        "Tenure": int(tenure),
        "Balance": float(balance),
        "NumOfProducts": int(num_of_products),
        "HasCrCard": has_cr_card,
        "IsActiveMember": is_active_member,
        "EstimatedSalary": float(estimated_salary),
        "Germany": is_germany,
        "Spain": is_spain,
        "Male": is_male,
    }

# Function to load the TensorFlow model
def load_model():
    model = tf.keras.models.load_model("churn_model.h5")
    return model

# Function to preprocess input data for model prediction
def preprocess_input_data(user_input):
    input_data = pd.DataFrame([user_input])
    return input_data

# Function to make predictions using the loaded model
def predict_churn(model, input_data):
    prediction = model.predict(input_data)
    return prediction

# Function to post-process churn predictions
def post_process_predictions(predictions):
    return [1 if element > 0.5 else 0 for element in predictions]

# Function to display the DataFrame
def display_data_frame(user_input):
    st.subheader("User Input:")
    df = pd.DataFrame([user_input])
    st.table(df)

# Streamlit App
def main():
    st.title("Churn Prediction App")
    st.write(
        "Welcome to the Churn Prediction App! Enter the customer details in the sidebar and see the predictions."
    )

    # Load the TensorFlow model
    churn_model = load_model()

    user_input = get_user_input()

    # Preprocess input data and make predictions
    input_data = preprocess_input_data(user_input)
    churn_prediction = predict_churn(churn_model, input_data)

    # Post-process churn predictions
    post_processed_prediction = post_process_predictions(churn_prediction)

    # Display user input
    display_data_frame(user_input)

    st.subheader("Churn Prediction:")
    if post_processed_prediction[0] == 1:
        st.warning("High Churn Risk!")
    else:
        st.success("Low Churn Risk.")

    # Optional: Display raw prediction value
    st.write(f"Post-processed Prediction Output: {post_processed_prediction[0]}")

if __name__ == "__main__":
    main()
