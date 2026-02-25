import pandas as pd
import pickle
from data_cleaning import clean_dataset
from model import train_and_save

def run_all():
    train_and_save() # Step 1: Train
    
    # Step 2: Load and Predict on Test
    model = pickle.load(open('model.pkl', 'rb'))
    encoders = pickle.load(open('encoders.pkl', 'rb'))
    features = pickle.load(open('features.pkl', 'rb'))
    target_le = pickle.load(open('target_le.pkl', 'rb'))
    
    test_df = pd.read_csv("data_set/test.csv")
    cleaned_test = clean_dataset(test_df)
    
    for col, le in encoders.items():
        if col in cleaned_test.columns:
            cleaned_test[col] = cleaned_test[col].map(lambda s: s if s in le.classes_ else le.classes_[0])
            cleaned_test[col] = le.transform(cleaned_test[col].astype(str))
            
    preds = model.predict(cleaned_test[features])
    test_df['Predicted_Score'] = target_le.inverse_transform(preds)
    test_df.to_csv("results.csv", index=False)
    print("All tasks complete! Final results in results.csv")

if __name__ == "__main__":
    run_all()