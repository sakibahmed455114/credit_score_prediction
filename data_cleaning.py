import pandas as pd
import re
import numpy as np

def convert_to_months(text):
    if pd.isna(text) or not isinstance(text, str): return 0
    numbers = re.findall(r'\d+', text)
    if len(numbers) == 2:
        return (int(numbers[0]) * 12) + int(numbers[1])
    return int(numbers[0]) if len(numbers) == 1 else 0

def clean_dataset(df):
    df = df.copy()
    # Drop IDs
    df = df.drop(['ID', 'Customer_ID', 'Name', 'SSN', 'Month'], axis=1, errors='ignore')
    
    # Fill Salary
    if 'Occupation' in df.columns and 'Monthly_Inhand_Salary' in df.columns:
        df['Monthly_Inhand_Salary'] = df['Monthly_Inhand_Salary'].fillna(
            df.groupby('Occupation')['Monthly_Inhand_Salary'].transform('median')
        )
    
    # Numeric Cleanup
    cols = ['Num_of_Delayed_Payment', 'Num_Credit_Inquiries', 'Amount_invested_monthly', 'Monthly_Balance']
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace('[^0-9.]', '', regex=True), errors='coerce').fillna(0)
            
    if 'Credit_History_Age' in df.columns:
        df['Credit_History_Months'] = df['Credit_History_Age'].apply(convert_to_months)
        df.drop('Credit_History_Age', axis=1, inplace=True)
        
    return df.replace(('_', '-'), 0)