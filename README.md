# ML-projects
# Calorie Predictor - Machine Learning Project
This project focuses on predicting the number of calories burned during physical activities using a Linear Regression model. The model takes into account various factors such as age, height, weight, heart rate, and body temperature to estimate calorie expenditure.

**Table of Contents**

1.Project Overview

2.Dataset

3.Steps Followed

4.Tools & Libraries

5.Results

6.Future Enhancements


## Project Overview
The goal of this project is to build a predictive model that estimates the number of calories burned based on physical attributes and activity data. The project involves data preprocessing, exploratory data analysis (EDA), feature engineering, model training, and evaluation.

## Dataset
The dataset used in this project contains the following features:

Gender: Male or Female

Age: Age of the individual

Height: Height of the individual (in cm)

Weight: Weight of the individual (in kg)

Duration: Duration of the activity (in minutes)

Heart Rate: Heart rate during the activity

Body Temp: Body temperature during the activity

Calories: Calories burned (target variable)

The dataset was preprocessed to handle missing values, encode categorical variables, and add new features like BMI.


## Steps Followed
**Data Preprocessing**:
- Loaded and cleaned the dataset.
- Encoded categorical variables (e.g., gender).
- Added new features like BMI (Body Mass Index).
**Exploratory Data Analysis (EDA)**:
- Conducted data visualization to understand relationships between variables.
- Analyzed correlations between features and the target variable (calories burned).
- Feature Engineering:
- Scaled and normalized the data for better model performance.
**Model Training**:
-Split the dataset into training and testing sets.
-Trained a Linear Regression model on the training data.
**Model Evaluation**:
- Evaluated the model using metrics like Mean Absolute Error (MAE), Mean Squared Error (MSE), and R2 Score.
- Achieved an R2 score of 0.96, indicating high accuracy.

## Tools & Libraries

**Python**

**Pandas**: For data manipulation and analysis.

**NumPy**: For numerical computations.

**Matplotlib and Seaborn**: For data visualization.

**Scikit-learn**: For model training and evaluation.

## Results
The Linear Regression model performed well with the following evaluation metrics:
**Mean Absolute Error (MAE)**: 8.03

**Mean Squared Error (MSE)**: 121.66

**R2 Score**: 0.96

The high R2 score indicates that the model is highly accurate in predicting calorie burn.

## Future Enhancements
- Incorporate additional features like activity type, step count, or external factors (e.g., temperature, sleep quality).
- Explore advanced machine learning models like Decision Trees, Random Forests, or Gradient Boosting for improved accuracy.
- Deploy the model as a web or mobile application for real-time calorie prediction.
