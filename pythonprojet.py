# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 21:26:41 2022

@author: Ben Mansour Célina
"""

import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
import numpy as np


def convert(pred):
    if pred ==0:
        return 'insufficient weight'
    elif pred ==1:
        return 'normal weight'
    elif pred ==2:
        return 'overweight level I'
    elif pred ==3:
        return 'overweight level II'
    elif pred ==4:
        return 'obesity level I'
    elif pred ==5:
        return 'obesity level II' 
    else :
        return 'obesity level III'



#Data

df=pd.read_csv("ObesityDataSet_raw_and_data_sinthetic.csv")

#Processing

from sklearn.preprocessing import LabelEncoder

encoder=LabelEncoder()
df_cleaned=df.copy()
df_cleaned["Gender"]=encoder.fit_transform(df["Gender"])
df_cleaned["family_history_with_overweight"]=encoder.fit_transform(df["family_history_with_overweight"])
df_cleaned["FAVC"]=encoder.fit_transform(df["FAVC"])
df_cleaned["SMOKE"]=encoder.fit_transform(df["SMOKE"])
df_cleaned["SCC"]=encoder.fit_transform(df["SCC"])
df_cleaned["MTRANS"]=encoder.fit_transform(df["MTRANS"])

df_cleaned["NObeyesdad"]=df_cleaned["NObeyesdad"].replace("Insufficient_Weight",0)
df_cleaned["NObeyesdad"]=df_cleaned["NObeyesdad"].replace("Normal_Weight",1)
df_cleaned["NObeyesdad"]=df_cleaned["NObeyesdad"].replace("Overweight_Level_I",2)
df_cleaned["NObeyesdad"]=df_cleaned["NObeyesdad"].replace("Overweight_Level_II",3)
df_cleaned["NObeyesdad"]=df_cleaned["NObeyesdad"].replace("Obesity_Type_I",4)
df_cleaned["NObeyesdad"]=df_cleaned["NObeyesdad"].replace("Obesity_Type_II",5)
df_cleaned["NObeyesdad"]=df_cleaned["NObeyesdad"].replace("Obesity_Type_III",6)
df_cleaned["CAEC"]=df_cleaned["CAEC"].replace("no",0)
df_cleaned["CAEC"]=df_cleaned["CAEC"].replace("Sometimes",1)
df_cleaned["CAEC"]=df_cleaned["CAEC"].replace("Frequently",2)
df_cleaned["CAEC"]=df_cleaned["CAEC"].replace("Always",3)
df_cleaned["CALC"]=df_cleaned["CALC"].replace("no",0)
df_cleaned["CALC"]=df_cleaned["CALC"].replace("Sometimes",1)
df_cleaned["CALC"]=df_cleaned["CALC"].replace("Frequently",2)
df_cleaned["CALC"]=df_cleaned["CALC"].replace("Always",3)

#Create model

data = pd.DataFrame(df_cleaned[["Age","Weight","Height","family_history_with_overweight", "FAVC","FCVC","FAF","CAEC","NObeyesdad"]])
target = "NObeyesdad"
x= data[["Age","Weight","Height","family_history_with_overweight","FAVC","FAF","FCVC","CAEC"]]
y= data[target]

minmax_scale = MinMaxScaler().fit(x)

X = minmax_scale.transform(x)

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X, y)

decision_tree=DecisionTreeClassifier( )
decision_tree.fit(X,y)

with open('intro.txt') as f:
    intro = f.read()


#Sidebar
st.sidebar.title("Obesity level prediction")
st.sidebar.text('Introduction')
st.sidebar.text(intro)


#Body

st.header('Try it yourself ! ')
age=st.text_input("Age","")
weight=st.text_input("Weight","")
height=st.text_input("Height","")
family_hist=st.selectbox("Family history with overweight",["yes","no"])
favc=st.selectbox("FAVC : Frequent consumption of high caloric food",["yes","no"])
fcvc=st.text_input("FCVC : Frequency of consumption of vegetables : never : 0 / sometimes : 1 / always : 2","")
caec=st.selectbox("CAEC : Consumption of food between meals",["no","sometimes","frequently","always"])
faf=st.text_input("FAF : Physical activity frequency : I do not have : 0 / 1 or 2 days : 1 / 2 or 4 days : 2 / 4 or 5 days : 3","")

bool_predict=st.button("Predict")

if bool_predict:
    age = int(age)
    weight = int(weight)
    height = float(height)
    fcvc = int(fcvc)
    faf = int(faf)
    if family_hist=="yes":
        family_hist=1
    else :
        family_hist = 0
    if favc=="yes":
        favc=1
    else :
        favc = 0
    if caec =='no':
        caec = 0
    elif caec =="sometimes":
        caec =1
    elif caec == "frequently":
        caec = 2
    else :
        caec = 3
    X_test = minmax_scale.transform(np.array([age,weight,height,family_hist,favc,fcvc,caec,faf]).reshape((1,-1)))
    pred_rf=random_forest.predict(X_test)
    pred_dt=decision_tree.predict(X_test)
    st.text("Random Forect prediction : " + convert(pred_rf[0]))
    st.text("Descision Tree prediction : " + convert(pred_dt[0]))
    
    
    
