import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit
streamlit.title("Sample data for model")
df = pd.read_csv("StudentsPerformance.csv")
streamlit.dataframe(df.head())
streamlit.title("input data for ml model")
columns=['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch','test_preparation_course','reading_score','writing_score']
data=[]
for i in columns[:5]:
  choice = streamlit.text_input(f'provide input for {i}','sample data')
  data.append(str(choice))
for i in columns[5:]:
  choice = streamlit.number_input(f'provide input for {i}','sample number')
  data.append(choice)
models = ['LinearRegression()','Lasso()','Ridge()','KNeighborsRegressor()','DecisionTreeRegressor()','RandomForestRegressor()','AdaBoostRegressor()']
model_selected=streamlit.selectbox("Pick ml model:",models) 

# pickled_model = pickle.load(open('model.pkl', 'rb'))
# pickled_model.predict(X_test)
  
  

