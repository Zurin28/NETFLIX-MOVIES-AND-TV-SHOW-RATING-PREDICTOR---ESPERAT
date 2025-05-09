from django.db import models

class TitanicPassenger(models.Model):
    GENDER_CHOICES = [
        (0, 'Male'),
        (1, 'Female'),
    ]
    
    EMBARKED_CHOICES = [
        ('C', 'Cherbourg'),
        ('Q', 'Queenstown'),
        ('S', 'Southampton'),
    ]
    
    CLASS_CHOICES = [
        (1, '1st Class'),
        (2, '2nd Class'),
        (3, '3rd Class'),
    ]
    
    pclass = models.IntegerField(choices=CLASS_CHOICES, verbose_name='Passenger Class')
    sex = models.IntegerField(choices=GENDER_CHOICES, verbose_name='Gender')
    age = models.FloatField(verbose_name='Age')
    sibsp = models.IntegerField(verbose_name='Siblings/Spouse Aboard')
    parch = models.IntegerField(verbose_name='Parents/Children Aboard')
    fare = models.FloatField(verbose_name='Fare')
    embarked = models.CharField(max_length=1, choices=EMBARKED_CHOICES, verbose_name='Port of Embarkation')
    survived = models.BooleanField(verbose_name='Survived', null=True)
    prediction_date = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = 'Titanic Passenger'
        verbose_name_plural = 'Titanic Passengers'

    def __str__(self):
        return f"Passenger (Class: {self.pclass}, Age: {self.age})"