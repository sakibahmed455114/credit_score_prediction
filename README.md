# Credit Score Classification System

A modular Machine Learning pipeline designed to clean financial data, train gradient-boosted models, and provide a modern web interface for real-time credit score predictions.

---

## 📂 Project Structure

| File | Description |
| :--- | :--- |
| **`data_cleaning.py`** | The "Engine." Contains logic to handle missing values, convert time-strings to integers, and clean non-numeric characters. |
| **`train_pipeline.py`** | The "Builder." Processes `train.csv`, trains an XGBoost classifier, and exports model artifacts (`.pkl`). |
| **`main.py`** | The "Manager." Orchestrates the full flow: Trains the model and then runs inference on `test.csv`. |
| **`app.py`** | The "UI." A modern Streamlit web dashboard for interactive manual predictions. |
| **`train.csv` / `test.csv`** | Your raw datasets. |

---

## 🛠️ Installation & Setup

### 1. Prerequisites
Ensure you have Python 3.9+ installed. You will need the following libraries:

```bash
pip install pandas numpy scikit-learn xgboost streamlit
