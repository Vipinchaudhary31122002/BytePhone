import streamlit as st
import pickle
import numpy as np
import json

# Load trained model
model = pickle.load(open('./model/bytephone-2025.pickle', 'rb'))

# Load column and mapping info
with open("./model/columns.json", "r") as f:
    columns_data = json.load(f)
    data_columns = columns_data["data_columns"]
    cat_mappings = columns_data.get("categorical_mappings", {})

# Load dropdown options generated from dataset
with open("./model/user_input.json", "r") as f:
    dropdown_data = json.load(f)

# UI Setup
st.set_page_config(page_title="📱 BytePhone Price Predictor")
st.title("📱 BytePhone 2025 - Mobile Price Prediction App")
st.write("Fill in the details to predict the estimated mobile price.")

# Separate features
numeric_features = data_columns[:5]  # Adjust this if needed
categorical_features = list(cat_mappings.keys())

# Input containers
user_input = []
st.subheader("🔢 Technical Specifications")

# Numeric feature inputs
for feature in numeric_features:
    options = dropdown_data.get(feature)
    if options:
        try:
            options = sorted(set(map(float, options)))
            selected = st.selectbox(f"{feature}", options)
        except:
            selected = st.selectbox(f"{feature}", options)
        user_input.append(float(selected))
    else:
        val = st.number_input(f"{feature}", value=64.0)
        user_input.append(val)

# Categorical feature inputs
st.subheader("🏷️ Choose Categories")
one_hot = np.zeros(len(data_columns))

st.subheader("🏢 Brand & Model Info")
brand_related = ["Company Name", "Model Name", "Processor"]
for feature in brand_related:
    if feature in dropdown_data:
        options = dropdown_data[feature]
        selected = st.selectbox(f"{feature}", options)
        col_name = f"{feature}_{selected}"
        if col_name in data_columns:
            idx = data_columns.index(col_name)
            one_hot[idx] = 1
        if feature in categorical_features:
            categorical_features.remove(feature)


for feature in categorical_features:
    options = dropdown_data.get(feature, [])
    selected = st.selectbox(f"{feature}", options)
    col_name = f"{feature}_{selected}"
    if col_name in data_columns:
        idx = data_columns.index(col_name)
        one_hot[idx] = 1

# Merge numeric features into the final input array
for i, val in enumerate(user_input):
    one_hot[i] = val


# Prediction
if st.button("💰 Predict Price"):
    prediction = model.predict([one_hot])[0]
    st.success(f"📱 Estimated Mobile Price: ${prediction:,.2f}")
