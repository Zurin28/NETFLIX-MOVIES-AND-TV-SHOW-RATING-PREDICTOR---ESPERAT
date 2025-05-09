import os
import pickle
import pandas as pd
import joblib
import numpy as np
from datetime import datetime
from django.shortcuts import render, redirect
from django.conf import settings
from django.contrib import messages
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth.decorators import login_required
from .forms import TitanicPredictionForm, ShowPredictionForm
from sklearn.feature_extraction.text import TfidfVectorizer
from .utils.netflix_predictor import NetflixRatingPredictor
import plotly.express as px
import plotly.utils
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Required for non-interactive backend
import io
import base64

# Load the model and vectorizer (you'll need to train these first)
try:
    model = joblib.load('shows/ml_models/netflix_rating_model.pkl')
    vectorizer = joblib.load('shows/ml_models/tfidf_vectorizer.pkl')
    label_encoder = joblib.load('shows/ml_models/label_encoder.pkl')
except:
    model = None
    vectorizer = None
    label_encoder = None

predictor = NetflixRatingPredictor()

def home_view(request):
    return render(request, 'shows/home.html')

def login_view(request):
    if request.user.is_authenticated:
        return redirect('dashboard')
    
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        
        if user is not None:
            login(request, user)
            return redirect('dashboard')
        else:
            messages.error(request, 'Invalid username or password.')
    
    return render(request, 'accounts/login.html')  # Changed template path

def register_view(request):
    if request.user.is_authenticated:
        return redirect('predict_show')
    
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('predict_show')
    else:
        form = UserCreationForm()
    
    return render(request, 'accounts/register.html', {'form': form})

def logout_view(request):
    logout(request)
    return redirect('login')

@login_required(login_url='login')
def predict_show(request):
    prediction = None
    if request.method == 'POST':
        try:
            show_type = request.POST.get('type')
            director = request.POST.get('director')
            duration = request.POST.get('duration')
            description = request.POST.get('description')
            
            # Make prediction
            result = predictor.predict(
                show_type,
                director,
                duration,
                description
            )
            
            if result['success']:
                prediction = {
                    'status': 'success',
                    'rating': result['rating'],
                    'confidence': result.get('confidence', 'N/A')
                }
            else:
                prediction = {
                    'status': 'error',
                    'message': result['message']
                }
            
        except Exception as e:
            prediction = {
                'status': 'error',
                'message': f"Error: {str(e)}"
            }
    
    return render(request, 'shows/predict_show.html', {
        'prediction': prediction
    })

@login_required(login_url='login')
def predict_titanic(request):
    prediction = None
    form = TitanicPredictionForm(request.POST if request.method == 'POST' else None)
    
    if request.method == 'POST' and form.is_valid():
        try:
            model_path = os.path.join(settings.BASE_DIR, 'shows', 'Model', 'ml_model.aziz')
            scaler_path = os.path.join(settings.BASE_DIR, 'shows', 'Model', 'scaler.aziz')
            features_path = os.path.join(settings.BASE_DIR, 'shows', 'Model', 'feature_names.aziz')
            
            # Load model, scaler and feature names
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            with open(features_path, 'rb') as f:
                feature_names = pickle.load(f)
            
            # Create base input data
            input_data = pd.DataFrame({
                'Pclass': [int(form.cleaned_data['pclass'])],
                'Sex': [int(form.cleaned_data['sex'])],
                'Age': [float(form.cleaned_data['age'])],
                'SibSp': [int(form.cleaned_data['sibsp'])],
                'Parch': [int(form.cleaned_data['parch'])],
                'Fare': [float(form.cleaned_data['fare'])],
            })
            
            # Add embarked columns with all zeros
            for col in feature_names:
                if col.startswith('Embarked_') and col not in input_data:
                    input_data[col] = 0
            
            # Set the selected embarked value to 1
            embarked = form.cleaned_data['embarked']
            input_data[f'Embarked_{embarked}'] = 1
            
            # Ensure columns are in the same order as training
            input_data = input_data[feature_names]
            
            # Scale and predict
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]
            messages.success(request, "Prediction completed successfully!")
            
        except FileNotFoundError:
            messages.error(request, "Model files not found. Please train the model first.")
        except Exception as e:
            messages.error(request, f"Error making prediction: {str(e)}")
    
    return render(request, 'shows/titanic_template.html', {
        'form': form,
        'prediction': prediction
    })

@login_required
def predict_netflix_rating(request):
    if request.method == 'POST':
        # Get form data
        title = request.POST.get('title')
        show_type = request.POST.get('type')
        director = request.POST.get('director')
        duration = request.POST.get('duration')
        description = request.POST.get('description')
        
        # Make prediction
        predicted_rating = predictor.predict(
            title, show_type, director, duration, description
        )
        
        return render(request, 'shows/predict_netflix.html', {
            'prediction': f'Predicted Rating: {predicted_rating}'
        })
    
    return render(request, 'shows/predict_netflix.html')

@login_required
def dashboard_view(request):
    try:
        data_file = os.path.join(settings.BASE_DIR, 'shows', 'data', 'netflix_titles.csv')
        df = pd.read_csv(data_file)
        
        stats = {
            'total_content': int(len(df)),
            'movies': int(len(df[df['type'] == 'Movie'])),
            'tv_shows': int(len(df[df['type'] == 'TV Show'])),
            'unique_ratings': int(df['rating'].nunique()),
        }
        
        # Rating Distribution Plot
        plt.figure(figsize=(10, 6))
        plt.style.use('dark_background')
        rating_counts = df['rating'].value_counts()
        plt.pie(rating_counts.values, 
                labels=rating_counts.index,
                autopct='%1.1f%%',
                colors=plt.cm.Pastel1(np.linspace(0, 1, len(rating_counts))))
        plt.title('Content Rating Distribution')
        
        # Save rating plot to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', 
                   facecolor='#2d2d2d', edgecolor='none')
        buffer.seek(0)
        rating_chart = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        # Yearly Content Plot
        plt.figure(figsize=(10, 6))
        yearly_data = df.groupby('release_year').size()
        plt.plot(yearly_data.index, yearly_data.values, 
                marker='o', color='#E50914', linewidth=2)
        plt.grid(True, alpha=0.2)
        plt.title('Content by Release Year')
        plt.xlabel('Year')
        plt.ylabel('Number of Titles')
        plt.xticks(rotation=45)
        
        # Save yearly plot to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight',
                   facecolor='#2d2d2d', edgecolor='none')
        buffer.seek(0)
        yearly_chart = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        context = {
            'stats': stats,
            'rating_chart': rating_chart,
            'yearly_chart': yearly_chart,
            'error': None
        }
        
    except FileNotFoundError:
        context = {
            'error': 'Dataset file not found. Please ensure netflix_titles.csv is in the shows/data directory.'
        }
    except Exception as e:
        context = {
            'error': f'Error processing dashboard data: {str(e)}'
        }
    
    return render(request, 'shows/dashboard.html', context)