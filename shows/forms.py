from django import forms

class ShowPredictionForm(forms.Form):
    title = forms.CharField(max_length=100, label="Show Title")
    genre = forms.ChoiceField(choices=[
        ('drama', 'Drama'),
        ('comedy', 'Comedy'),
        ('thriller', 'Thriller'),
        ('documentary', 'Documentary')
    ])
    rating = forms.FloatField(min_value=0, max_value=10, label="Rating")
    description = forms.CharField(widget=forms.Textarea, label="Description")

class TitanicPredictionForm(forms.Form):
    GENDER_CHOICES = [
        (0, 'Male'),
        (1, 'Female'),
    ]
    
    EMBARKED_CHOICES = [
        ('C', 'Cherbourg'),
        ('Q', 'Queenstown'),
        ('S', 'Southampton'),
    ]
    
    pclass = forms.ChoiceField(
        choices=[(1, '1st Class'), (2, '2nd Class'), (3, '3rd Class')],
        label='Passenger Class'
    )
    sex = forms.ChoiceField(choices=GENDER_CHOICES, label='Gender')
    age = forms.FloatField(label='Age', min_value=0, max_value=120)
    sibsp = forms.IntegerField(label='Siblings/Spouse Aboard', min_value=0)
    parch = forms.IntegerField(label='Parents/Children Aboard', min_value=0)
    fare = forms.FloatField(label='Fare', min_value=0)
    embarked = forms.ChoiceField(choices=EMBARKED_CHOICES, label='Port of Embarkation')