import pandas as pd
import pickle
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from data_cleaning import clean_dataset

def train_and_save():
    df = pd.read_csv("data_set/train.csv")
    df = clean_dataset(df)
    
    encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        if col != 'Credit_Score':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            
    target_le = LabelEncoder()
    y = target_le.fit_transform(df['Credit_Score'])
    X = df.drop('Credit_Score', axis=1)
    
    model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, num_class=3)
    model.fit(X, y)
    
    # Save artifacts
    pickle.dump(model, open('model.pkl', 'wb'))
    pickle.dump(encoders, open('encoders.pkl', 'wb'))
    pickle.dump(target_le, open('target_le.pkl', 'wb'))
    pickle.dump(X.columns.tolist(), open('features.pkl', 'wb'))
    print("Models trained and saved!")

if __name__ == "__main__":
    train_and_save()