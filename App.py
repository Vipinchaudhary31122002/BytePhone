import streamlit as st
import pickle
import numpy as np
import json

# Load trained model
model = pickle.load(open('./model/model.pickle', 'rb'))

# Load column and mapping info
with open("./model/columns.json", "r") as f:
    columns_data = json.load(f)
    data_columns = columns_data["data_columns"]
    cat_mappings = columns_data.get("categorical_mappings", {})

# Load input fields
with open("./model/user_input.json", "r") as f:
    input_fields = json.load(f)

# Streamlit UI
st.set_page_config(page_title="📱 BytePhone")
st.title("📱 BytePhone 2025 - Mobile Price Prediction App")
st.write("Fill in the details to predict the estimated mobile price.")

# Input form
with st.form("user_input_form"):
    st.markdown("### 📝 Phone Specifications")
    user_input = {}

    for field, values in input_fields.items():
        field_label = field.replace("_", " ").title()
        if len(set(values)) == 2:
            user_input[field] = st.selectbox(field_label, sorted(set(values)))
        else:
            user_input[field] = st.selectbox(field_label, sorted(values))

    submit_btn = st.form_submit_button("💰 Predict Price")

# Convert user input to model input format
if submit_btn:
    input_vector = [0] * len(data_columns)

    for field in user_input:
        value = user_input[field]
        key = f"{field}_{value}" if field in cat_mappings else field
        if key in data_columns:
            idx = data_columns.index(key)
            input_vector[idx] = 1 if field in cat_mappings else float(value)

    # Make prediction
    prediction = model.predict([input_vector])[0]
    st.success(f"📱 Estimated Mobile Price: ${prediction:,.2f}")
