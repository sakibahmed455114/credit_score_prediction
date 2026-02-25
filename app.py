import streamlit as st
import pandas as pd
import pickle
from data_cleaning import clean_dataset

st.set_page_config(page_title="Credit AI", layout="wide")
st.title("Real-time Credit Scorer")

@st.cache_resource
def load_assets():
    return (pickle.load(open('model.pkl', 'rb')), 
            pickle.load(open('encoders.pkl', 'rb')), 
            pickle.load(open('target_le.pkl', 'rb')), 
            pickle.load(open('features.pkl', 'rb')))

try:
    model, encoders, target_le, features = load_assets()
    
    with st.sidebar:
        st.header("Input Customer Data")
        occ = st.selectbox("Occupation", list(encoders['Occupation'].classes_))
        income = st.number_input("Annual Income", 1000, 500000, 50000)
        history = st.text_input("Credit History Age", "5 Years and 2 Months")
        btn = st.button("Predict Score")

    if btn:
        input_df = pd.DataFrame([{"Occupation": occ, "Annual_Income": income, "Credit_History_Age": history}])
        for f in features:
            if f not in input_df.columns: input_df[f] = 0
            
        cleaned = clean_dataset(input_df)
        for col, le in encoders.items():
            if col in cleaned.columns: cleaned[col] = le.transform(cleaned[col].astype(str))
            
        res = model.predict(cleaned[features])[0]
        st.balloons()
        st.success(f"### The predicted Credit Score is: **{target_le.inverse_transform([res])[0]}**")
except:
    st.error("Run 'python main.py' first to train the model!")