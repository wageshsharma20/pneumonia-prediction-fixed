import os
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    SMOTE = None

app = Flask(__name__)

# Global variables to hold the trained components
model = None
scaler = None
train_columns = None

def train_model():
    global model, scaler, train_columns
    
    # Ensure dataset exists
    dataset_path = 'clinical_pneumonia_dataset.csv'
    if not os.path.exists(dataset_path):
        print("Dataset not found. Please ensure clinical_pneumonia_dataset.csv is present.")
        return

    df = pd.read_csv(dataset_path)

    # Convert binary features to 0/1
    # We must handle both 'Yes'/'No' and potential existing 0/1
    binary_cols = ['fever', 'tachycardia', 'crackles']
    for col in binary_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0, 1: 1, 0: 0}).fillna(0).astype(int)

    # Create target variable
    df['target'] = df['true_label'].map({'pneumonia': 1, 'normal': 0}).fillna(0).astype(int)

    X = df[['fever', 'tachycardia', 'crackles', 
            'oxygen_saturation', 'wbc_count', 'chest_xray_result']]
    y = df['target']

    # One-hot encoding for categorical variables
    X_encoded = pd.get_dummies(X, columns=['chest_xray_result'])
    train_columns = X_encoded.columns

    # Feature scaling for continuous variables
    scaler = StandardScaler()
    X_encoded[['oxygen_saturation', 'wbc_count']] = scaler.fit_transform(
        X_encoded[['oxygen_saturation', 'wbc_count']]
    )

    # Handle class imbalance if SMOTE is available
    if SMOTE:
        try:
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X_encoded, y)
        except Exception as e:
            print(f"SMOTE failed: {e}, falling back to original data")
            X_resampled, y_resampled = X_encoded, y
    else:
        X_resampled, y_resampled = X_encoded, y

    # Train model
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_resampled, y_resampled)
    print("Model trained successfully.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No JSON received'})

        # Extract and format input data
        input_data = {
            'fever': [1 if data.get('fever') == 'Yes' else 0],
            'tachycardia': [1 if data.get('tachycardia') == 'Yes' else 0],
            'crackles': [1 if data.get('crackles') == 'Yes' else 0],
            'oxygen_saturation': [float(data.get('oxygen_saturation', 0))],
            'wbc_count': [float(data.get('wbc_count', 0))],
            'chest_xray_result': [data.get('xray_result', 'Normal')]
        }

        df_input = pd.DataFrame(input_data)
        df_input_encoded = pd.get_dummies(df_input, columns=['chest_xray_result'])

        # Align columns with training data
        for col in train_columns:
            if col not in df_input_encoded.columns:
                df_input_encoded[col] = 0
        
        df_input_encoded = df_input_encoded[train_columns]

        # Scale continuous features
        df_input_encoded[['oxygen_saturation', 'wbc_count']] = scaler.transform(
            df_input_encoded[['oxygen_saturation', 'wbc_count']]
        )

        # Make prediction
        prob = model.predict_proba(df_input_encoded)[0, 1]
        is_high_risk = prob >= 0.35

        return jsonify({
            'success': True,
            'prediction': 'High Risk of Pneumonia' if is_high_risk else 'Low Risk (Clear)',
            'probability': f"{prob * 100:.1f}%"
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# Train the model when the application starts
train_model()

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
