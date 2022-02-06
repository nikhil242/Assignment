#!/usr/bin/env python
# coding: utf-8

# In[118]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[119]:


df = pd.read_csv('Dataset.csv')


# In[120]:


df.head()


# In[121]:


df.describe()


# In[122]:


df.info()


# In[123]:


p=df.Loan_Status.value_counts()
p.head()


# In[124]:


p.plot.pie(labels = ("Y", "N"),autopct = "%.2f%%")


# In[125]:


p.plot.bar(color=('red','blue')).set(xticklabels = ["Y", "N"])


# In[138]:


sns.distplot(df.LoanAmount) 


# In[140]:


sns.boxplot(df.ApplicantIncome)


# In[126]:


df.isna().sum()


# In[127]:


df.dropna(inplace=True, axis=0) #axis value 0, means its a row. axis value 1 means its a column


# In[128]:


df.isna().sum()


# In[129]:


df.shape


# In[130]:


# Male = 0, Female = 1
df['Gender'] = df['Gender'].replace({'Male':0, 'Female':1,'unknown' : 2})
# Yes = 1, No = 0
df['Married'] = df['Married'].replace({'Yes' :1, 'No': 0, 'unknown':2})
# Graduate = 1, Not Graduate = 0
df['Education'] = df['Education'].replace ({'Graduate' : 1, 'Not Graduate' : 0})
# Yes: 1
# No : 0
# unknown:2
df['Self_Employed'] = df['Self_Employed'].replace ({'Yes': 1,'No' : 0, 'unknown':2})
df['Property_Area'] = df['Property_Area'].replace ({'Semiurban': 1,'Urban' : 0, 'Rural':2})
df['Loan_Status'] = df['Loan_Status'].replace({'Y':1, 'N':0})
df['Dependents'] = df['Dependents'].replace({'3+':3})


# In[131]:


df.head()


# In[132]:


from sklearn.model_selection import train_test_split

X = df.iloc[:,1:-1]
y = df.iloc[:, -1]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=100)
X.head() 


# In[133]:


from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
LR = KNeighborsClassifier(n_neighbors=10)


# In[134]:


#fiting the model
LR.fit(X_train, y_train)

#prediction
y_pred = LR.predict(X_test)

#Accuracy
print("Accuracy ", LR.score(X_test, y_test)*100)


# In[ ]:


Gen= input("Input Gender 1 for Male 0 for Female")
Marr= input("If marrried Input 1 for Yes and 0 for No")
Depen= input("Depedents present ? \n Input 1, 2 or 3+ in case of more than 3")
Edu= input ("Education level \n Input 0 for Not Graduate 0 and 1 for Graduate ")
SelfEmp= input("Self employed ? \nInput 1 for Yes 0 for No")
AppInc= input("Enter Applicant income")
CoApInc=input("Enter co Applicant income")
LoAmt=input("Enter loan amount")
LoAmtTerm=input("Enter loan amount term")
Crehis=input("Enter credit history")
PropAre=input("Enter property area1 for urban and 0 for rural")
X_actual_values=[Gen,Marr,Depen,Edu,SelfEmp,AppInc,CoApInc,LoAmt,LoAmtTerm,Crehis,PropAre]
X_actual_values


# In[ ]:


X_actual_values=np.array(X_actual_values).astype('int16')
X_actual_values=X_actual_values.reshape(1,11)
X_actual_values=pd.DataFrame(X_actual_values)
X_actual_values.columns=(['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
'Loan_Amount_Term', 'Credit_History', 'Property_Area'])
y_actual_pred=LR.predict(X_actual_values)
print('Should the person be given a loan ? \n1 for yes 0 for no. \n As per KNN the answer is =',y_actual_pred)


# In[ ]:





# In[ ]:





# In[ ]:




