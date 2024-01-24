import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import streamlit
streamlit.title("Sample data for model")
df = pd.read_csv("StudentsPerformance.csv")
streamlit.dataframe(df.head())
streamlit.title("input data for ml model")
columns=['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch','test_preparation_course','reading_score','writing_score']
# for i in columns[:5]:
#   choice = streamlit.text_input(f'provide input for {i}','sample data')
#   data.append(str(choice))
gender=streamlit.selectbox("provide input for Gender:", ['female', 'male'])
re=streamlit.selectbox("provide input for race_ethnicity:", ['group B', 'group C', 'group A', 'group D', 'group E'])
PLE=streamlit.selectbox("Pick some parental_level_of_education:",["bachelor's degree", 'some college', "master's degree","associate's degree", 'high school', 'some high school'])
Lunch=streamlit.selectbox("Pick some Lunch:",['standard', 'free/reduced'])
tpc=streamlit.selectbox("Pick some test_preparation_course:",['none', 'completed'])
data=[gender,re,PLE,Lunch,tpc]
for i in columns[5:]:
  choice = streamlit.number_input(f'provide input for {i}')
  data.append(choice)
models = ['LinearRegression()','Lasso()','Ridge()','KNeighborsRegressor()','DecisionTreeRegressor()','RandomForestRegressor()','AdaBoostRegressor()']
model_selected=streamlit.selectbox("Pick ml model:",models) 
streamlit.text('Selected data :')
x={}
for i in range(len(data)):
  x[columns[i]] = [data[i]]
data=pd.DataFrame(data=x)
streamlit.dataframe(data)
preprocessing_model = pickle.load(open('preprocesser.pkl', 'rb'))
data=preprocessing_model.transform(data)
ml_model = pickle.load(open(f'{model_selected}.pkl', 'rb'))
  
  

