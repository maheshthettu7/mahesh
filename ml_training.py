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
# for i in columns[:5]:
#   choice = streamlit.text_input(f'provide input for {i}','sample data')
#   data.append(str(choice))
gender=streamlit.selectbox("provide input for Gender:", ['female', 'male'])
re=streamlit.selectbox("provide input for race_ethnicity:", ['group B', 'group C', 'group A', 'group D', 'group E'])
PLE=streamlit.selectbox("Pick some parental_level_of_education:",["bachelor's degree", 'some college', "master's degree","associate's degree", 'high school', 'some high school'])
Lunch=streamlit.selectbox("Pick some Lunch:",['standard', 'free/reduced'])
tpc=streamlit.selectbox("Pick some test_preparation_course:"['none', 'completed'])
data.append([gender,re,PLELunch,tpc])
for i in columns[5:]:
  choice = streamlit.number_input(f'provide input for {i}')
  data.append(choice)
models = ['LinearRegression()','Lasso()','Ridge()','KNeighborsRegressor()','DecisionTreeRegressor()','RandomForestRegressor()','AdaBoostRegressor()']
model_selected=streamlit.selectbox("Pick ml model:",models) 
streamlit.text(data)
# pickled_model = pickle.load(open('model.pkl', 'rb'))
# pickled_model.predict(X_test)
  
  

