# Netflix Show Predictor - Django Project

## Overview

This project provides a simple Django web interface to predict whether a Netflix show is "Highly Rated" based on its rating. Users can submit a show’s name, genre, rating, and description, and receive a prediction based on the rating. The interface simulates a model prediction based on the rating value.

## Form Fields

1. **Show Name:**  
   - **Field Type:** Text input  
   - **Description:** The name of the Netflix show you want to analyze.

2. **Genre:**  
   - **Field Type:** Text input  
   - **Description:** The genre of the show (e.g., Action, Drama, Comedy).

3. **Rating:**  
   - **Field Type:** Number input (0-10)  
   - **Description:** The rating score of the show, from 0 to 10. This is used to predict if the show is highly rated.

4. **Description:**  
   - **Field Type:** Textarea  
   - **Description:** A short description of the show, providing context or storyline information.

## How the UI Interacts with the Model

1. The user fills out the form with the required show information (name, genre, rating, description).
2. Upon form submission, the backend processes the rating value:
   - If the rating is greater than 7, the prediction is that the show is "Highly Rated."
   - If the rating is 7 or lower, the prediction is "Not Highly Rated."
3. The result (prediction) is displayed on the same page once the form is successfully submitted.

## How to Use

1. Clone the repository and set up your Django project.
2. Run `python manage.py migrate` to set up the database.
3. Start the Django development server using `python manage.py runserver`.
4. Visit `http://127.0.0.1:8000/predict/` in your browser to use the prediction form.

## Future Enhancements

- Replace the simulated prediction with a real machine learning model for more accurate predictions.
- Add more advanced prediction features based on additional show data.
