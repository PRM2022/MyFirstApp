
#### 
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from PIL import Image


image = Image.open('LinkedIn_Graphic.png')

st.image(image, caption='Source: Google Images 2022')


st.title('Probability that YOU are a LinkedIn User')





#### Education Selectbox####
educ = st.selectbox("Education Level", 
             options = ["Less than High School Diploma",
                        "High School - Incomplete",
                        "High School - Graduate",
                        "Some College - No Degree",
                        "Two-Year Associate's Degree",
                        "Four Year Bachelors Degree",
                        "Some Post-Graduate or Professional Schooling - No Degree",
                        "Post-Graduate or Professional Degree"
                        ])

#st.write(f"Education (pre-conversion): {educ}")

#st.write("**Convert Selection to Numeric Value**")

if educ == "Less than High School Diploma":
    educ = 1
elif educ == "High School - Incomplete":
    educ = 2
elif educ == "High School - Graduate":
    educ = 3
elif educ == "Some College - No Degree":
    educ = 4
elif educ == "Two-Year Associate's Degree":
    educ = 5
elif educ == "Four Year Bachelors Degree":
    educ = 6
elif educ == "Some Post-Graduate or Professional Schooling - No Degree":
    educ = 7
else: 
    educ = 8
#st.write(f"Education (post-conversion): {educ}")


####INCOME####
income = st.selectbox("Gross Household Income Level", 
             options = ["Less than $10,000",
                        "$10,000 to under $20,000",
                        "$20,000 to under $30,000",
                        "$30,000 to under $40,000",
                        "$40,000 to under $50,000",
                        "$50,000 to under $75,000",
                        "$75,000 to under $100,000",
                        "$100,000 to under $150,000",
                        "$150,000 or more?"
                        ])

#st.write(f"Income (pre-conversion): {income}")


if income == "Less than $10,000":
    income = 1
elif income == "$10,000 to under $20,000":
    income = 2
elif income == "$20,000 to under $30,000":
    income = 3
elif income == "$30,000 to under $40,000":
    income = 4
elif income == "$40,000 to under $50,000":
    income = 5
elif income == "$50,000 to under $75,000":
    income = 6
elif income == "$75,000 to under $100,000":
    income = 7
elif income == "$100,000 to under $150,000":
    income = 8
else: 
    income = 9
#st.write(f"Income (post-conversion): {income}")



####Parent####

child = st.selectbox("Parental Status",
            options= ["Yes",
                      "No",
                        ])
#st.write(f"Parental Status Selected: {child}")

if child == "Yes":
    child = 1
else:
    child = 0 

####ENDPARENT####


####MaritalStatus####

marriage = st.selectbox("Marital Status",
            options= ["Yes",
                      "No",
                        ])
#st.write(f"Marital Status Selected: {marriage}")

if marriage == "Yes":
    marriage = 1
else:
    marriage = 0 

####ENDMaritalStatus####   



####Gender####
gender = st.selectbox("Gender",
            options= ["Male",
                      "Female",
                        ])
#st.write(f"Gender Selected: {gender}")

if gender == "Female":
    gender = 1
else:
    gender = 0 


####ENDGender####

####Age####

age = st.number_input('Enter Your Age',
                min_value= 1,
                max_value= 99,
                value=30)
#st.write("Your Age is:", age)

####EndAge####


s = pd.read_csv(r"social_media_usage.csv")

def clean_sm(x):
   
    x = pd.DataFrame(data=np.where(x == 1, 1, 0), columns =x.columns)
                     



ss = pd.DataFrame({
    
    "sm_li":np.where(s["web1h"] == 1, 1, 0),
    
    "income":np.where(s["income"] > 9,np.nan,s["income"]),
    
    "education":np.where(s["educ2"]> 8,np.nan,s["educ2"]),
    
    "parent":np.where(s["par"] == 1,1,0),
    
    "married": np.where(s["marital"] ==1,1,0),
    
    "female": np.where(s["gender"] ==2,1,0),
    
    "age":np.where(s["age"] >98, np.nan,s["age"])})


ss = ss.dropna()


y = ss["sm_li"]
x = ss[["income","education","parent","married","female","age"]]

  
x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    stratify=y,       # same number of target in training & test set
                                                    test_size=0.2,    # hold out 20% of data for testing
                                                    random_state=153)


lr = LogisticRegression(class_weight="balanced")
lr.fit(x_train, y_train)  

y_pred = lr.predict(x_test)


person_1 = pd.DataFrame({ #### Change variable name
    
    "income":[income],
    
    "education":[educ],
    
    "parent":[child],
    
    "married": [marriage],
    
    "female": [gender],
    
    "age":[age]
    
})
prob=(lr.predict_proba(person_1))[0][1]

prob = round(prob*100,1)

st.markdown(f"Results {prob}%")


###############  Dr. Lyon's Code ###################################


#### Create label (called sent) from TextBlob polarity score to use in summary below

if prob > 50:

    label = "Probably"

elif prob < 49.9

    label = "Probably Not"

else:

    label = "Who knows?!"

   

##### Show results



#### Print sentiment score, label, and language

st.markdown(f"You are ***{label}*** a LinkedIn User" )



#### Create sentiment gauge

fig = go.Figure(go.Indicator(

    mode = "gauge+number",

    value = prob,

    title = {'text': f"Probability that YOU are a LinkedIn user: {label}"},

    gauge = {'axis': {"range": [0, 100], 'tickwidth': 1, 'tickcolor': "black"},
             'bar':{"color":"red"},
             'bgcolor': "white",
             'borderwidth': 2,
             'bordercolor': "black", 
            'steps': [
                {"range": [0, 50], "color":"grey"},
                {"range": [50.01, 100], "color":"darkblue"}],
               # {"range": [0,1 ], "color":"black"}
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': prob}}))  
            

st.plotly_chart(fig)
