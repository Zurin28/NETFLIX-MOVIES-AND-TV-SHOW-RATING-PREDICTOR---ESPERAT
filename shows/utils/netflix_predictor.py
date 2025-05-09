import numpy as np
import joblib
import os
from django.conf import settings

class NetflixRatingPredictor:
    def __init__(self):
        self.model = None
        self.tfidf = None
        self.label_encoder = None
        self.scaler = None
        self.load_models()
    
    def load_models(self):
        try:
            models_dir = os.path.join(settings.BASE_DIR, 'shows', 'ml_models')
            self.model = joblib.load(os.path.join(models_dir, 'logistic_model.joblib'))
            self.tfidf = joblib.load(os.path.join(models_dir, 'tfidf_vectorizer.joblib'))
            self.label_encoder = joblib.load(os.path.join(models_dir, 'rating_label_encoder.joblib'))
            self.scaler = joblib.load(os.path.join(models_dir, 'scaler.joblib'))
        except Exception as e:
            print(f"Error loading models: {e}")
    
    def extract_numeric_features(self, type_val, director, duration, description):
        """Extract and prepare numeric features"""
        return np.array([[
            float(type_val),
            float(duration),
            float(len(str(director))),
            float(len(description)),
            float(len(description.split()))
        ]])
    
    def predict(self, type_val, director, duration, description):
        if not all([self.model, self.tfidf, self.label_encoder, self.scaler]):
            return {
                'success': False,
                'message': "Model not loaded. Please ensure model files are present."
            }
        
        try:
            # 1. Get TF-IDF features for description
            tfidf_features = self.tfidf.transform([description]).toarray()
            
            # 2. Get and scale numeric features
            numeric_features = self.extract_numeric_features(
                type_val, director, duration, description
            )
            scaled_numeric = self.scaler.transform(numeric_features)
            
            # 3. Combine features
            combined_features = np.hstack([scaled_numeric, tfidf_features])
            
            # 4. Make prediction
            prediction = self.model.predict(combined_features)[0]
            
            # 5. Convert prediction back to rating
            predicted_rating = self.label_encoder.inverse_transform([prediction])[0]
            
            return {
                'success': True,
                'rating': predicted_rating,
                'confidence': 'high'  # You can add model confidence later
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f"Error making prediction: {str(e)}"
            }