## 📁 01. wine quality prediction 
* Implement linear regression using the least squares method
* Use learning curves plot to understand whether the linear model is overfitting or underfitting
* Extend model with polynomial basis expansions and regularizations (Ridge and Lasso) using Sklearn
* Use k-fold cross validation to obtain the optimal hyper-parameters for the models
## 📁 02. car price prediction 
* Prepare data and Exploratory data analysis (EDA)
* Define baseline model 
* Use linear regression for predicting price
* Evaluating the model with RMSE
* Feature engineering
* Tuning the model : Regularization 
* Using the best model

## Frequently used concept 
### encoding categorical data
Categorical variables are typically strings, and pandas identifies them as object types. These variables need to be converted to a numerical form because ML models can interpret only numerical features. It is possible to incorporate certain categories from a feature, not necessarily all of them. This transformation from categorical to numerical variables is known as One-Hot encoding.

## 🪄 Side note 
### ✨ Useful pandas function for Data preparation 
* pd.read_csv(<file_path_string>) - read csv files
* df.head() - take a look of the dataframe
* df.columns - retrieve colum names of a dataframe
* df.columns.str.lower() - lowercase all the letters
* df.columns.str.replace(' ', '_') - replace the space separator
* df.dtypes - retrieve data types of all features
* df.index - retrieve indices of a dataframe

### ✨ Useful pandas, Matplotlib and seaborn function for EDA
* df[col].unique() - returns a list of unique values in the series
* df[col].nunique() - returns the number of unique values in the series
* df.isnull().sum() - retunrs the number of null values in the dataframe
* %matplotlib inline - assure that plots are displayed in jupyter notebook's cells
* sns.histplot() - show the histogram of a series

### ✨ Useful pandas and Numpy function for spliting datasets 
* df.iloc[] - returns subsets of records of a dataframe, being selected by numerical indices
* df.reset_index() - restate the orginal indices
* del df[col] - eliminates target variable
* np.arange() - returns an array of numbers
* np.random.shuffle() - returns a shuffled array
* np.random.seed() - set a seed
