#!/usr/bin/env python
# coding: utf-8

# In[1]:


#jaiiii sri ram


# In[2]:


#steps
'''
1.input raw data 
2.tokenizing 
3.textcleaning 
   - remove punctuations
   - remove stopwords
   - stemming/lemmatization
4.vectorization
5.modelling
6.accuracy and final model
'''


# # BUSINESS PROBLEM UNDERSTANDING 

# In[3]:


#Performing sentimental analysis on the input data and saying whether the feedback of the students is positive or not 
#application - applying on the feedback data of the students on the lecturer and analying their data instead of manually reading each and every feedback


# In[4]:


import pandas as pd
import seaborn as sns
import matplotlib as plt


# In[5]:


df = pd.read_csv("C:\\Users\\Sricharan Reddy\\Downloads\\feedback dataset.csv")


# In[6]:


df.head()


# In[7]:


df.info()


# In[8]:


#here the first column is a text column in which the feedback is written and sentiment is the output column of dataset


# In[9]:


import nltk
import re


# In[10]:


from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# In[11]:


ps = PorterStemmer()


# In[12]:


corpus = []
for i in range(len(df)):
    rp = re.sub("[^a-zA-Z]"," ",df['text'][i])
    rp = rp.lower()
    rp = rp.split()
    rp = [ps.stem(word) for word in rp if not  word in set(stopwords.words('english'))]
    rp = " ".join(rp)
    corpus.append(rp)


# In[13]:


corpus[:5]


# In[14]:


#working fine here we have done removing punctuations , stopwords , stemming lessgooo


# In[15]:


from sklearn.feature_extraction.text import CountVectorizer


# In[16]:


cv = CountVectorizer()


# In[17]:


x = cv.fit_transform(corpus).toarray()


# In[18]:


y = df['sentiment']


# In[19]:


from sklearn.model_selection import train_test_split


# In[20]:


x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2,random_state = 9)


# In[ ]:





# # Modelling 

# ## 1.Naive Bayes classifier

# In[21]:


from sklearn.naive_bayes import MultinomialNB


# In[22]:


nbmodel = MultinomialNB()


# In[23]:


nbmodel.fit(x_train,y_train)


# In[24]:


y_pred_train  = nbmodel.predict(x_train)


# In[25]:


y_pred_test = nbmodel.predict(x_test)


# In[26]:


from sklearn.metrics import accuracy_score


# In[27]:


accuracy_score(y_train,y_pred_train)


# In[28]:


accuracy_score(y_test,y_pred_test)


# In[29]:


nbmodel.score(x_test,y_test)


# ## 2.XGBOOST ALGORITHM 

# In[30]:


from xgboost import XGBClassifier


# In[31]:


model = XGBClassifier()
model.fit(x_train,y_train)


# In[32]:


y_pred_train  = model.predict(x_train)
y_pred_test = model.predict(x_test)


# In[33]:


from sklearn.metrics import accuracy_score


# In[34]:


accuracy_score(y_train,y_pred_train)


# In[35]:


accuracy_score(y_test,y_pred_test)


# In[36]:


model.score(x_test,y_test)


# # 3.Logistic regression

# In[37]:


from sklearn.linear_model import LogisticRegression


# In[38]:


model = LogisticRegression()
model.fit(x_train,y_train)


# In[39]:


y_pred_train  = model.predict(x_train)
y_pred_test = model.predict(x_test)


# In[40]:


print(accuracy_score(y_train,y_pred_train))
print(accuracy_score(y_test,y_pred_test))


# In[41]:


model.score(x_test,y_test)


# # To make a new prediction : - 

# In[42]:


#for predicting for a new input you have to do all the steps of the preprocessing before model u can't just give it directly


# In[59]:


new_feedback = "teaching is excellent and the way of explanation is very good "


# In[60]:


#new_feedback = "teaching is not good the way of explanation is bad and ambiguious "


# In[61]:


corpus = []


# In[62]:


rp = re.sub("[^a-zA-Z]"," ",new_feedback)
rp = rp.lower()
rp = rp.split()
rp = [ps.stem(word) for word in rp if not  word in set(stopwords.words('english'))]
rp = " ".join(rp)
corpus.append(rp)


# In[63]:


x_new = cv.transform(corpus).toarray()


# In[64]:


x_new


# In[65]:


ans = model.predict(x_new)


# In[66]:


if(ans==1):
    print("Positive opinion")
else:
    print("Negative opinion")


# In[67]:


#scr


# In[ ]:




