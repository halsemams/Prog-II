# Final Project
## Mike Halsema
### 12/8/22

import streamlit as st 

import pandas as pd

import numpy as np

import altair as alt

import plotly.express as px

import seaborn as sns

import os

import statsmodels.formula.api as smf



from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# ***

#### Q1

os.chdir("C:/Users/halse/Documents/project")

s = pd.read_csv("social_media_usage.csv")

print(s.shape)

# ***

#### Q2

def clean_sm(x):
    x= np.where(x == 1, 
                            1,
                            0)
    return x
    

df = pd.DataFrame({'a':[1,2,2],
                    'b':[2,1,2]})


# df

df["c"]= clean_sm(df["a"])

# df

# ***

#### Q3


ss = pd.DataFrame({
    "income":np.where(s["income"] > 9, np.nan,
                      s["income"]),
    "education":np.where(s["educ2"] > 8, np.nan,
                      s["educ2"]),
    "parent":np.where(s["par"] > 2, np.nan,
                          np.where(s["par"] == 1, 1, 0)),
     "married":np.where(s["marital"] > 6, np.nan,
                          np.where(s["marital"] < 2, 1, 0)),  
    "female":np.where(s["gender"] > 2, np.nan,
                          np.where(s["gender"] < 2, 1, 0)),
    "age":np.where(s["age"] > 98, np.nan,
                      s["age"])          
    })


# ss

#Add new Linkedin column
clean_sm(s["web1h"])
ss["sm_li"] = clean_sm(s["web1h"])


#check new column
# ss

#check NaN
check_nan = ss.isnull().values.any()
# check_nan

ss=ss.dropna()

#check Nan
check_nan = ss.isnull().values.any()
# check_nan

##Apprears to be wrong data type
# ss.dtypes

#change data types
ss['income'] = ss['income'].astype(np.int64)
ss['education'] = ss['education'].astype(np.int64)
ss['parent'] = ss['parent'].astype(np.int64)
ss['married'] = ss['female'].astype(np.int64)
ss['female'] = ss['female'].astype(np.int64)
ss['age'] = ss['age'].astype(np.int64)

# ss

# ss.dtypes

# print(ss.shape)

#exploatory analysis
#frequency 
pd.crosstab(ss["sm_li"],columns="count")

#frequency 
pd.crosstab(ss["sm_li"],ss["female"],normalize = "index")

#frequency 
pd.crosstab(ss["sm_li"],ss["married"],normalize = "index")

#frequency 
pd.crosstab(ss["sm_li"],ss["parent"],normalize = "index")

#frequency 
pd.crosstab(ss["sm_li"],ss["education"],normalize = "index")

#frequency 
pd.crosstab(ss["sm_li"],ss["income"],normalize = "index")

#correlation
ss["age"].corr(ss["education"])

#correlation
ss["age"].corr(ss["income"])

#correlation
ss["income"].corr(ss["education"])

#regression 
formula = smf.ols("income~education + age", data = ss)
model = formula.fit()
model.summary()

alt.Chart(ss.groupby(["age", "income"], as_index=False)["sm_li"].mean()).\
mark_circle().\
encode(x="age",
      y="sm_li",
      color="income:N")

# ***

#### Q4

# + Target : sm_li
#     + Has linkedin account (=1)
#     + Does not have linked in account (=0)

# + Features :    
#     + income (1 low -> 9 high )
#     + education (1 no hs -> 8 professional degree )
#     + parent (bianary)
#     + married (bianary)
#     + female (bianary)
#     + age (numberic)

# Target (y) and feature(s) selection (X)
y = ss["sm_li"]
X = ss[["income", "education", "parent", "married", "female", "age"]]

# ***

#### Q5

# Split data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    stratify=y,       # same number of target in training & test set
                                                    test_size=0.2,    # hold out 20% of data for testing
                                                    random_state=987) # set for reproducibility

# X_train contains 80% of the data and contains the features used to predict the target when training the model. 
# X_test contains 20% of the data and contains the features used to test the model on unseen data to evaluate performance. 
# y_train contains 80% of the the data and contains the target that we will predict using the features when training the model. 
# y_test contains 20% of the data and contains the target we will predict when testing the model on unseen data to evaluate performance.

# ***

#### Q6

# Initialize algorithm 
lr = LogisticRegression()

lr.fit(X_train, y_train)

#### Q7

# Make predictions using the model and the testing data
y_pred = lr.predict(X_test)

# Get other metrics with classification_report
print(classification_report(y_test, y_pred))

# ***

#### Q8

# Compare those predictions to the actual test data using a confusion matrix (positive class=1)

#confusion_matrix(y_test, y_pred)

cm = pd.DataFrame(confusion_matrix(y_test, y_pred),
            columns=["Predicted negative", "Predicted positive"],
            index=["Actual negative","Actual positive"]).style.background_gradient(cmap="PiYG")
# cm

# ***

#### Q9 

## recall: TP/(TP+FN)
recall = 41/(41+42)
# recall

## precision: TP/(TP+FP)
precision = 41/(41+28)
# precision


f1_score = 2 * (precision * recall) / (precision + recall)
# f1_score

# ***

#### Q10

newdata = pd.DataFrame({
    "income": [9, 6,4],
    "education": [2, 8, 4],
    "parent": [0, 0, 1],
    "married": [1, 0, 1],
    "female": [0, 1, 1],
    "age": [25, 45, 33]
})

# newdata

newdata["prediction_sm_li"] = lr.predict(newdata)

# newdata

person = [8,7,0,1,1,42]

# Predict class, given input features
predicted_class = lr.predict([person])

# Generate probability of positive class (=1)
probs = lr.predict_proba([person])

# Print predicted class and probability
print(f"Predicted class: {predicted_class[0]}") # 0=not pro-environment, 1=pro-envronment
print(f"Probability that this person is pro-environment: {probs[0][1]}")

entdata = pd.DataFrame({
    "income": [],
    "education": [],
    "parent": [],
    "married": [],
    "female": [],
    "age": []
})

 
st.markdown("# Linkedin User Predictor")
 


with st.sidebar:
    income = st.selectbox("Income level", 
             options = ["Less than $10,000",
                        "$10,000 to under $20,000",
                        "20,000to under $30,000",
                         "30,000 to under $40,000",
                        "40,000 to under $50,000",
                        "50,000 to under $75,000",
                        "75,000 to under $100,000",
                        "100,000 to under $150,000",
                        "$150,000 or more"])
    education = st.selectbox("Education level", 
             options = ["No high school",
                        "High school not complete",
                        "High school diploma or equivelent",
                         "Some college, but no degree",
                        "Associates degree",
                        "Bachelor's degree",
                        "Some postgraduate schooling, but not postgraduate degree",
                        "Pots graduate degree"])
    parent = st.selectbox("Are you a parent of a child under 18 living in your home?", 
             options = ["Yes",
                        "No"])
    married = st.selectbox("Marital Status", 
             options = ["Married",
                        "Living with a partner",
                        "Divorced",
                        "Separated",
                        "Widowed",
                        "Neven been married"])
    gender = st.selectbox("Gender", 
             options = ["Female",
                       "Male",
                        "Non-Bianary"])
    age = st.slider(label="Age", 
          min_value=12,
          max_value=98,
          value=32)
    

st.write(f"Income: {income}")

# st.write("**Convert Selection to Numeric Value**")

if income == "Less than $10,000":
    income = 1
elif income == "$10,000 to under $20,000":
    income = 2
elif income == "20,000to under $30,000":
    eincome = 3
elif income == "30,000 to under $40,000":
    income = 4
elif income == "40,000 to under $50,000":
    income = 5
elif income == "50,000 to under $75,000":
    income = 6
elif income == "75,000 to under $100,000":
    income = 7
elif income == "100,000 to under $150,000":
    income = 8
else:
    income = 9
    
# st.write(f"Income (post-conversion): {income}")

### Education



st.write(f"Education: {education}")

# st.write("**Convert Selection to Numeric Value**")

if education == "No high school":
    education = 1
elif education == "High school not complete":
    education = 2
elif education == "High school diploma or equivelen":
    education = 3
elif education == "Some college, but no degree":
    education= 4
elif education == "Associates degree":
    education= 5
elif education == "Bachelor's degree":
    education = 6
elif education == "Some postgraduate schooling, but not postgraduate degree":
    education = 7
else:
    education = 8

# st.write(f"Education (post-conversion): {education}")


### Parent

  

st.write(f"Parent?: {parent}")

# st.write("**Convert Selection to Numeric Value**")

if parent== "Yes":
    parent = 1
else:
    parent = 0

# st.write(f"Parent (post-conversion): {parent}")


### married



st.write(f"Marital Status: {married}")

# st.write("**Convert Selection to Numeric Value**")

if married== "Married":
    married = 1
else:
    married = 0

# st.write(f"Married (post-conversion): {married}")

###Gender


st.write(f"Gender: {gender}")

# st.write("**Convert Selection to Numeric Value**")

if gender== "Female":
    gender = 1
else:
    gender = 0

# st.write(f"Gender (post-conversion): {gender}")




st.write("Age: ", age)

# ########## Example 2

# num1 = st.slider(label="Enter a number", 
#           min_value=1,
#           max_value=9,
#           value=7)

# num2 = st.slider(label="Enter a number",
#           min_value=1,
#           max_value=100,
#           value=50)

# num3 = st.slider(label="Enter a number", 
#           min_value=1,
#           max_value=8,
#           value=5)

# st.write("Your numbers: ", num1, num2, num3)

# num_sum = num1+num2+num3

# st.write("Your numbers: ", num1, num2, num3, "sum to", num_sum)


selected_option = (income,education,parent,married,gender,age)

predicted_class1 = lr.predict([selected_option])

predicted_class1

probs1 = lr.predict_proba([selected_option])

probs1 = lr.predict_proba([selected_option])

probs1

