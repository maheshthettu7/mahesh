import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor
import warnings
streamlit.title("Sample data for model")
df = pd.read_csv("StudentsPerformance.csv")
streamlit.dataframe(df.head())
streamlit.title("input data for ml model")
columns=['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch','test_preparation_course','reading_score','writing_score']
data=[]
for i in columns[0:5]:
  choice = streamlit.text_input(f'provide input for {i}','sample data')
  data.append(str(choice))
  
  

