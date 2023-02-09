# Logistic Regression (Binary classification)
* Logistic regression is a model to predict probability of the occurence of events
* Core idea of logistic regression is same as linear regression( learning w, which satisfies wx+w0), but returning values between 0 and 1 using sigmoid function. So we need to find weight using gradient descent methods
* In general, threshold is 0.5 but it can be tunned based on the problem
* feature selection is possible by interpreting weights of each feature
## 📁 01. LR vs NB

## 📁 02. Churn prediction project

The project aims to identify customers that are likely to churn or stoping to use a service. Each customer has a score associated with the probability of churning. Considering this data, the company would send an email with discounts or other promotions to avoid churning. 

The ML strategy applied to approach this problem is binary classification, which for one instance can be expressed as: 

<p align="center">
    <img src="https://render.githubusercontent.com/render/math?math=\large g\left(x_{i}\right) = y_{i}"/>
</p>

In the formula, yi is the model's prediction and belongs to {0,1}, being 0 the negative value or no churning, and 1 the positive value or churning. The output corresponds to the likelihood of churning. 

In brief, the main idea behind this project is to build a model with historical data from customers and assign a score of the likelihood of churning. 

For this project, we used a [Kaggle dataset](https://www.kaggle.com/blastchar/telco-customer-churn). 
