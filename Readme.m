Project Description
Online Store Purchase Prediction

This project involves training a machine learning model to predict whether a user will make a purchase at an online store, using data from user sessions.

The data comes from a dataset published in 2019 and includes features related to user behavior during their sessions at the store. The target variable is the "Revenue" column, which indicates whether the user made a purchase or not.
Objectives

    Prepare and normalize the dataset
    Train a logistic regression model
    Evaluate the model's performance

Content

    Data Preparation:
        Remove the features Month, Browser, and OperatingSystems
        Convert boolean values to numeric
        Apply one-hot encoding to the variables Region, TrafficType, and VisitorType
        Split the dataset into training and test sets
        Normalize the data

    Model Training:
        Create and train a logistic regression model

    Model Evaluation:
        Calculate and print the accuracy on the training and test sets
        Calculate and print the confusion matrix
