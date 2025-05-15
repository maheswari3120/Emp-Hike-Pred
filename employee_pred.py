# 🎯# 📦 Essential Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# 📂 Load Data
path = 'WA_Fn-UseC_-HR-Employee-Attrition.csv'
df = pd.read_csv(path)

# 📌 Drop Unnecessary Columns
to_drop = ['Attrition', 'DistanceFromHome', 'Over18', 'MaritalStatus', 'RelationshipSatisfaction',
           'JobSatisfaction', 'EnvironmentSatisfaction', 'Gender', 'EmployeeCount',
           'EmployeeNumber', 'StandardHours', 'OverTime']
df = df.drop(columns=to_drop)

# 📌 One-hot Encoding for Categorical Variables
categorical_columns = ['BusinessTravel', 'Department', 'EducationField', 'JobRole']
df = pd.get_dummies(df, columns=categorical_columns)

# 📌 Remove Outliers (Filtering)
df = df[(df['MonthlyIncome'] <= 180000)]
df = df[(df['NumCompaniesWorked'] <= 8)]
df = df[(df['TotalWorkingYears'] <= 35)]

# 🎯 Categorize 'PercentSalaryHike' into 3 Classes
bins = [0, 12, 18, 25]
labels = [0, 1, 2]  # 0=Low, 1=Medium, 2=High
df['SalaryHikeCategory'] = pd.cut(df['PercentSalaryHike'], bins=bins, labels=labels)

# 📌 Define Features and Target (Reduced Inputs)
selected_features = ['Age', 'MonthlyIncome', 'TotalWorkingYears', 'YearsAtCompany',
                     'PerformanceRating', 'JobLevel', 'NumCompaniesWorked']

X = df[selected_features]
y_class = df['SalaryHikeCategory']
y_reg = df['PercentSalaryHike']

# 📊 Split Dataset
X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
    X, y_class, y_reg, test_size=0.2, random_state=42)

# 🌳 Random Forest Classifier (Tuned)
clf = RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_split=4,
                             min_samples_leaf=2, random_state=54)
clf.fit(X_train, y_class_train)

# Save the classifier
with open('random_forest_classifier.pkl', 'wb') as f:
    pickle.dump(clf, f)

y_class_pred = clf.predict(X_test)

# 🎯 Classification Accuracy
acc = accuracy_score(y_class_test, y_class_pred)
print(f"\n✅ Classification Accuracy: {acc*100:.2f}%")

# 📊 Confusion Matrix
cm = confusion_matrix(y_class_test, y_class_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Low', 'Medium', 'High'], yticklabels=['Low', 'Medium', 'High'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# 📊 Classification Report
print(classification_report(y_class_test, y_class_pred, target_names=['Low', 'Medium', 'High']))

# 🌳 Random Forest Regressor (Tuned)
reg = RandomForestRegressor(n_estimators=300, max_depth=12, min_samples_split=4,
                            min_samples_leaf=2, random_state=42)
reg.fit(X_train, y_reg_train)
y_reg_pred = reg.predict(X_test)

# 🎯 Mean Absolute Error
mae = mean_absolute_error(y_reg_test, y_reg_pred)
print(f"📊 Mean Absolute Error (Salary Hike % prediction): {mae:.2f}")

# ✍️ User Input (Reduced to 7 Fields)
print("\n💻 Enter Employee Details:")

user_input = {}
for col in selected_features:
    user_input[col] = float(input(f"Enter {col}: "))

user_df = pd.DataFrame([user_input])

# 🎯 Predict for User Input
pred_class = clf.predict(user_df)[0]
pred_reg = reg.predict(user_df)[0]

# 💰 Calculate New Monthly Salary
current_income = user_input['MonthlyIncome']
new_income = current_income * (1 + pred_reg / 100)

# 🎯 Display Results
categories = {0: 'Low', 1: 'Medium', 2: 'High'}
print(f"\n💼 Predicted Salary Hike Category: {categories[int(pred_class)]}")
print(f"📈 Predicted Percent Salary Hike: {pred_reg:.2f}%")
print(f"💰 Predicted New Monthly Income: ₹{new_income:.2f}")
mae = mean_absolute_error(y_reg_test, y_reg_pred)
print(f"📊 Mean Absolute Error (Salary Hike % prediction): {mae:.2f}")

# ✍️ User Input (Reduced to 7 Fields)
print("\n💻 Enter Employee Details:")

user_input = {}
for col in selected_features:
    user_input[col] = float(input(f"Enter {col}: "))

user_df = pd.DataFrame([user_input])

# 🎯 Predict for User Input
pred_class = clf.predict(user_df)[0]
pred_reg = reg.predict(user_df)[0]

# 💰 Calculate New Monthly Salary
current_income = user_input['MonthlyIncome']
new_income = current_income * (1 + pred_reg / 100)

# 🎯 Display Results
categories = {0: 'Low', 1: 'Medium', 2: 'High'}
print(f"\n💼 Predicted Salary Hike Category: {categories[int(pred_class)]}")
print(f"📈 Predicted Percent Salary Hike: {pred_reg:.2f}%")
print(f"💰 Predicted New Monthly Income: ₹{new_income:.2f}")