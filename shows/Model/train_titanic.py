import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def train_model():
    # Get current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load and preprocess data
    train_data = pd.read_csv(os.path.join(current_dir, 'train.csv'))
    
    # Select and process features
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    df = train_data[features].copy()
    
    # Handle missing values
    df['Age'].fillna(df['Age'].mean(), inplace=True)
    df['Embarked'].fillna('S', inplace=True)
    
    # Encode categorical variables
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    
    # Create dummies for Embarked
    embarked_dummies = pd.get_dummies(df['Embarked'], prefix='Embarked')
    df = pd.concat([df.drop('Embarked', axis=1), embarked_dummies], axis=1)
    
    # Save feature names
    feature_names = df.columns.tolist()
    
    # Prepare training data
    X = df
    y = train_data['Survived']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    model = LogisticRegression(random_state=42)
    model.fit(X_scaled, y)
    
    # Save model, scaler and feature names
    with open(os.path.join(current_dir, 'ml_model.aziz'), 'wb') as f:
        pickle.dump(model, f)
    with open(os.path.join(current_dir, 'scaler.aziz'), 'wb') as f:
        pickle.dump(scaler, f)
    with open(os.path.join(current_dir, 'feature_names.aziz'), 'wb') as f:
        pickle.dump(feature_names, f)
    
    print("Model training completed successfully!")

if __name__ == '__main__':
    train_model()