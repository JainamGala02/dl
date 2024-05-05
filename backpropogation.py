#!/usr/bin/env python
# coding: utf-8

# **Komal Mahesh Chitnis**
# 
# **Moodle Id: 20102068**
# 
# **Subject: Deep Learning**
# 
# **Experiment No. 05**
# 
# **To implement backpropagation using the Keras library.**
# 

# In[2]:


import pandas as pd
import tensorflow
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense


# In[4]:


df = pd.read_csv('/content/drive/MyDrive/placementdataset.csv')


# In[5]:


df.head()


# In[6]:


df.shape


# In[7]:


df.duplicated().sum()


# In[8]:


df = df.drop_duplicates()


# In[9]:


df.duplicated().sum()


# In[10]:


df.shape


# In[11]:


X = df[['CGPA', 'Profile_Score']]
Y = df[['Placed']]


# In[12]:


X


# In[13]:


Y


# In[14]:


model = Sequential()
model.add(Dense(4, input_dim=2, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# In[15]:


model.summary()


# In[16]:


model.get_weights()


# In[17]:


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[18]:


history = model.fit(X, Y, epochs=100, batch_size = 1, verbose = 1)


# In[20]:


loss, accuracy = model.evaluate(X, Y)
print(f'Loss:{loss}, Accuracy:{accuracy}')


# In[23]:


import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()


# In[24]:


import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.title('Model Acccuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()


# In[25]:


import numpy as np
new_data = np.array([[8.1, 6.1]])
prediction = model.predict(new_data)


# In[26]:


prediction


# In[27]:


prediction_binary = (prediction > 0.5).astype(int)
print("Prediction:", prediction_binary)

