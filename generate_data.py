import pandas as pd
import numpy as np

def generate_synthetic_data(n_samples=200):
    np.random.seed(42)
    
    data = {
        'fever': np.random.choice(['Yes', 'No'], n_samples),
        'tachycardia': np.random.choice(['Yes', 'No'], n_samples),
        'crackles': np.random.choice(['Yes', 'No'], n_samples),
        'oxygen_saturation': np.random.uniform(85, 100, n_samples),
        'wbc_count': np.random.uniform(4000, 20000, n_samples),
        'chest_xray_result': np.random.choice(['Normal', 'Opacity', 'Infiltrate', 'Consolidation', 'Effusion'], n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Simple logic for target
    score = (df['fever'] == 'Yes').astype(int) * 2
    score += (df['oxygen_saturation'] < 92).astype(int) * 3
    score += (df['wbc_count'] > 11000).astype(int) * 2
    score += (df['chest_xray_result'] != 'Normal').astype(int) * 4
    
    df['true_label'] = np.where(score >= 5, 'pneumonia', 'normal')
    df.to_csv('clinical_pneumonia_dataset.csv', index=False)
    print("Synthetic dataset created: clinical_pneumonia_dataset.csv")

if __name__ == "__main__":
    generate_synthetic_data()
