# Support Vector Regression

In this project I will do a regression using the Support Vector Machine algorithm.
In machine learning, support-vector machines (SVMs, also support-vector networks) are supervised learning models with associated learning algorithms that analyze data for classification and regression analysis.
SVMs are one of the most robust prediction methods, being based on statistical learning frameworks.
The objective of the support vector machine algorithm is to find a hyperplane in an N-dimensional space(N — the number of features) that distinctly classifies the data points.
To separate the two classes of data points, there are many possible hyperplanes that could be chosen. Our objective is to find a plane that has the maximum margin, i.e the maximum distance between data points of both classes. Maximizing the margin distance provides some reinforcement so that future data points can be classified with more confidence.

# Dataset 
the dataset i use is a list of car prices in india.
consists of 12 columns and 5953 rows. the following is an explanation of each column:
Name: name of car

Location: located in india

Year: production year

Kilometers_Driven: how many kilometers have car traveled

Fuel_Type: type of fuel 

Owner_Type: new car or used

Transmission: manual or automatic

Mileage_kmpl: how far the car goes in 1 liter

Engine_cc: engine cc

Power_bhp: power 

Seats: how many seat

Price: the price i predict

# Explanation:
## Import Package
import common package:

**import numpy as np

**import pandas as pd


**from sklearn.model_selection import train_test_split

**from sklearn.pipeline import Pipeline

**from sklearn.compose import ColumnTransformer


**from jcopml.pipeline import num_pipe, cat_pipe

**from jcopml.utils import save_model, load_model

**from jcopml.plot import plot_missing_value

**from jcopml.feature_importance import mean_score_decrease


## Import Dataset
which i have explained before, the dataset has a column index called ID

## Mini Exploratory Data Analysis
I always work on data science projects with simple think so that I can benchmark.
Using a simple model to benchmark. And most of the time it's more efficient and sometimes find a good one. but at the beginning I did mini Exploratory Data Analysis.
because **i focus more on the algorithm**

In the dataset there is data that has no values ​​in several columns namely Engine_CC, Power_bhp, and Seats, but I leave it as is because I think it is an important feature. i will predict the price of the car.
I only removed the Name column which is the name of the car. Actually I can use Name for feature engineering. But it takes more effort to extract, and usually I do that when I don't get good score.

## Dataset Splitting
split the data into X, and y

X = all columns except the target column.

y = 'Price' as target

test_size = 0.2 (which means 80% for train, and 20% for test)

## Training
In the Training step there are 3 main things that I specify.

First, **the preprocessor**: here the columns will be grouped into numeric and categoric.

included in the numeric column are: 'Year', 'Kilometers_Driven', 'Mileage_kmpl', 'Engine_CC', 'Power_bhp', 'Seats'.

and in the categoric column are: 'Location', 'Fuel_Type', 'Transmission', 'Owner_Type'.
in categorical column I do encoding with onehot encoder

second, **pipeline**: contains the preprocessor as 'prep' which I defined earlier, and the algorithm as 'algo' which in this case I use Support Vector Regressor(SVR).

and third, tuning with **Grid Search**: in this case I use the tuning recommendations (gsp.svm_params) that often occur in many cases. but does not rule out hyperparameter tuning if the model results are not good.
with cross validation = 3.

### results and scalling
too bad the model score resulting from the tuning is so bad.
SVM is also very distance dependent, so I added scaling.

There are several types of scaling, including standard, minmax, and robust. for this time I use robust scaling.

as we can see in the #Scalling can help SVR and KNN on the notebook.
we can see the score increased very much. in the end we have to try everything. in the end the data will choose what kind of model it will use.

# Polynomial Features
in the last attempt, I put Polynomials into the model. 
here the tuning parameter that i used:

**{'prep__numeric__poly__degree': [1, 2, 3]**,

**'prep__numeric__poly__interaction_only': [True, False],**
 
 **'algo__gamma': array([1.e-03, 1.e-02, 1.e-01, 1.e+00, 1.e+01, 1.e+02, 1.e+03]),**
 
 **'algo__C': array([1.e-03, 1.e-02, 1.e-01, 1.e+00, 1.e+01, 1.e+02, 1.e+03])}**
 
 it just a recommendation, you can make your own grid parameters.
