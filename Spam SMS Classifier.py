#!/usr/bin/env python
# coding: utf-8

# In[2]:


#importing Lib/packages
import numpy as np
import pandas as pd


# In[3]:


#import dataset
dataset = pd.read_csv("SMSSpamCollection", sep='\t', names=['label', 'message'])


# In[4]:


dataset


# In[5]:


dataset.info()


# In[6]:


dataset.describe()


# In[7]:


#replacing ham and spam with 0 and 1
dataset['label']= dataset['label'].map({'ham': 0, 'spam': 1})


# In[8]:


dataset


# In[9]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[10]:


#countplot for spam vs ham as imbalanced dataset
plt.figure(figsize=(8,8))
g = sns.countplot(x="label", data=dataset)
p = plt.title('Countplot For Spam vs Ham as imbalanced dataset')
p = plt.xlabel('Is the SMS Spam?')
p = plt.ylabel('Count')


# In[11]:


# Handling imbalanced dataset using Oversampling
only_spam = dataset[dataset["label"] == 1]


# In[12]:


only_spam


# In[13]:


print("No of Spam SMS:", len(only_spam))
print("No of Ham SMS:", len(dataset) - len(only_spam))


# In[14]:


count = int((dataset.shape[0] - only_spam.shape[0])/only_spam.shape[0])


# In[15]:


count


# In[16]:


for i in range(0,count-1):
    dataset = pd.concat([dataset, only_spam]) 
dataset.shape 


# In[17]:


#countplot for spam vs ham as balanced dataset
plt.figure(figsize=(8,8))
g = sns.countplot(x="label", data=dataset)
p = plt.title('Countplot For Spam vs Ham as balanced dataset')
p = plt.xlabel('Is the SMS Spam?')
p = plt.ylabel('Count')


# In[18]:


dataset['word_count'] = dataset['message'].apply(lambda x:len(x.split()))


# In[19]:


dataset


# In[20]:


plt.figure(figsize=(12,6))

#(1,1)
plt.subplot(1,2,1)
g = sns.histplot(dataset[dataset["label"] == 0].word_count, kde=True)
p = plt.title('Distribution of word_count for Ham SMS')

#(1,2)
plt.subplot(1,2,2)
g = sns.histplot(dataset[dataset["label"] == 1].word_count, color="red", kde=True)
p = plt.title('Distribution of word_count for Spam SMS')

plt.tight_layout()
plt.show()


# In[21]:


#Creating new feature of containing currency symbols
def currency(data):
    currency_symbols = ['$', '€', '£', '¥', '₹']
    for i in currency_symbols:
        if i in data:
            return 1
    return 0


# In[22]:


dataset["contains_currency_symbols"] = dataset["message"].apply(currency)


# In[23]:


dataset


# In[24]:


#Countplot for contains_currency_symbols
plt.figure(figsize=(8,8))
g = sns.countplot(x='contains_currency_symbols', data=dataset, hue = "label")
p = plt.title('Countplot for Containing currency symbol')
p = plt.xlabel('Does SMS contains any currency symbol?')
p = plt.ylabel('count')
p = plt.legend(labels=["Ham", "Spam"], loc = 9)


# In[25]:


# Creating a new feature of containing numbers
def number(data):
    for i in data:
        if ord(i) >= 48 and ord(i) <= 57:
            return 1
    return 0    


# In[26]:


dataset["contains_number"] = dataset["message"].apply(number)


# In[27]:


dataset


# In[28]:


# Countplot for containing numbers
plt.figure(figsize=(8,8))
g = sns.countplot(x='contains_number', data=dataset, hue = "label")
p = plt.title('Countplot for Containing numbers')
p = plt.xlabel('Does SMS contains any number?')
p = plt.ylabel('count')
p = plt.legend(labels=["Ham", "Spam"], loc = 9)


# In[29]:


# Data cleaning
import nltk
import re
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# In[30]:


corpus = []
wnl = WordNetLemmatizer()

for sms in list(dataset.message):
    message = re.sub(pattern = '[^a-zA-Z]', repl=' ', string=sms) #Filtering out special characters and numbers
    message = message.lower()
    words = message.split()
    filtered_words = [word for word in words if word not in set(stopwords.words('english'))]
    lemm_words = [wnl.lemmatize(word) for word in filtered_words]
    message = ' '.join(lemm_words)
    
    corpus.append(message)


# In[32]:


corpus


# In[33]:


# creating the bag of words model
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features = 500)
vectors = tfidf.fit_transform(corpus).toarray()
feature_names = tfidf.get_feature_names_out()


# In[34]:


X = pd.DataFrame(vectors, columns = feature_names)
y = dataset['label']


# In[35]:


from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix


# In[36]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[37]:


X_train


# In[38]:


# Naive Bayes model
from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
cv =  cross_val_score(mnb, X, y, scoring= 'f1', cv=10)
print(round(cv.mean(),3))
print(round(cv.std(),3))


# In[39]:


mnb.fit(X_train, y_train)
y_pred = mnb.predict(X_test)


# In[41]:


print(classification_report(y_test, y_pred))


# In[42]:


cm = confusion_matrix(y_test, y_pred)


# In[46]:


plt.figure(figsize=(8,8))
axis_labels = ['ham', 'spam']
g = sns.heatmap(data=cm, xticklabels=axis_labels, yticklabels=axis_labels, annot = True, fmt='g', cbar_kws = {"shrink":0.5}, cmap='Blues')
p = plt.title("Confusion Matrix of Multinomial Naive Bays Model")
p = plt.xlabel('Actual Values')
p = plt.ylabel('Predicted Values')


# In[47]:


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
cv1 = cross_val_score(dt, X, y, scoring='f1', cv=10)
print(round(cv1.mean(),3))
print(round(cv1.std(),3))


# In[48]:


dt.fit(X_train, y_train)
y_pred1 = dt.predict(X_test)


# In[49]:


print(classification_report(y_test, y_pred1))


# In[50]:


cm = confusion_matrix(y_test, y_pred1)


# In[51]:


plt.figure(figsize=(8,8))
axis_labels = ['ham', 'spam']
g = sns.heatmap(data=cm, xticklabels=axis_labels, yticklabels=axis_labels, annot = True, fmt='g', cbar_kws = {"shrink":0.5}, cmap='Blues')
p = plt.title("Confusion Matrix of Multinomial Naive Bays Model")
p = plt.xlabel('Actual Values')
p = plt.ylabel('Predicted Values')


# In[70]:


def predict_spam(sms):
    message = re.sub(pattern = '[^a-zA-Z]', repl=' ', string=sms) #Filtering out special characters and numbers
    message = message.lower()
    words = message.split()
    filtered_words = [word for word in words if word not in set(stopwords.words('english'))]
    lemm_words = [wnl.lemmatize(word) for word in filtered_words]
    message = ' '.join(lemm_words)
    temp = tfidf.transform([message]).toarray()
    return mnb.predict(temp)


# In[ ]:




