# Decision Tree
* decision tree learns by reducing the impurity of the dataset
* impurity can be computed by Gini index or Entropy 
* decision tree algorithm : code 

# Random Forest
* emsemble methods that uses several decision tree
* several sub trees can be made with randomly choosen features 

## 📁 Scoring credit status with tree models 
Dataset: https://github.com/gastonstat/CreditScoring

### 1. Data load 
* Download the data from the given link.

### 2. Data preprocessing
* Reformat categorical columns (status, home, marital, records, and job) by mapping with appropriate values.
* Replace the maximum value of income, assests, and debt columns with NaNs.
* Replace the NaNs in the dataframe with 0 (will be shown in the next lesson).
* Extract only those rows in the column status who are either ok or default as value.
* Split the data into training/validation/test data
* Prepare target variable status by converting it from categorical to binary, where 0 represents ok and 1 represents default.
* split the data into feature X and label y 

### 3. Find the best model
#### 3.1 Decision Tree
#### 3.2 Random Forest
#### 3.3. Gradient Boosting
