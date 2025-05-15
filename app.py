from flask import Flask, render_template, request
import numpy as np
import pickle

# Load the classifier model
with open('random_forest_classifier.pkl', 'rb') as f:
    clf = pickle.load(f)

app = Flask(__name__)

# Mapping categories to average percent hikes
category_to_percent = {
    0: 10,   # Low
    1: 16,   # Medium
    2: 23.22  # High
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form inputs
        age = float(request.form['Age'])
        monthly_income = float(request.form['MonthlyIncome'])
        total_working_years = float(request.form['TotalWorkingYears'])
        years_at_company = float(request.form['YearsAtCompany'])
        performance_rating = float(request.form['PerformanceRating'])
        job_level = float(request.form['JobLevel'])
        num_companies_worked = float(request.form['NumCompaniesWorked'])

        # Create input array
        features = np.array([[age, monthly_income, total_working_years,
                              years_at_company, performance_rating,
                              job_level, num_companies_worked]])

        # Predict category
        pred_class = clf.predict(features)[0]
        categories = {0: 'Low', 1: 'Medium', 2: 'High'}
        hike_category = categories[int(pred_class)]

        # Simulate predicted percent hike
        pred_percent = category_to_percent[int(pred_class)]

        # Calculate new salary
        new_income = monthly_income * (1 + pred_percent / 100)

        return render_template('index.html',
                               prediction_text=f"Predicted Hike Category: {hike_category}",
                               percent_text=f"Predicted Percent Salary Hike: {pred_percent:.2f}%",
                               income_text=f"Predicted New Monthly Income: ₹{new_income:.2f}")

if __name__ == '__main__':
    app.run(debug=True)
