#!/usr/bin/env python
# coding: utf-8

# **Experiment 4**
# 
# Name: Komal Chitnis
# 
# Class: BE-A
# 
# Moodle id: 20102068
# 

# In[1]:


import tensorflow
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten


# In[2]:


(x_train, y_train),(X_test, y_test) = keras.datasets.mnist.load_data()


# In[3]:


x_train.shape


# In[4]:


x_train


# In[5]:


x_train[0]


# In[6]:


X_test.shape


# In[7]:


y_train


# In[9]:


import matplotlib.pyplot as plt
plt.imshow(x_train[0])


# In[10]:


plt.imshow(x_train[1])


# In[11]:


plt.imshow(x_train[9999])


# In[12]:


x_train = x_train/255
X_test = X_test/255


# In[13]:


x_train[0]


# In[15]:


model = Sequential()

model.add(Flatten(input_shape = (28,28)))
model.add(Dense(128,activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))

model.summary()


# In[17]:


model.compile(loss ='sparse_categorical_crossentropy',optimizer = 'Adam', metrics= ['accuracy'])*


history = model.fit(x_train, y_train, epochs = 10, validation_split = 0.2)


# In[18]:


model.predict(X_test)


# In[19]:


y_prob = model.predict(X_test)


# In[20]:


y_prob.argmax(axis =1)


# In[21]:


y_pred = y_prob.argmax(axis = 1)


# In[22]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)


# In[23]:


X_test[0]


# In[24]:


plt.imshow(X_test[0])


# In[25]:


plt.imshow(X_test[1])


# In[26]:


plt.imshow(X_test[2])


# In[27]:


model.predict(X_test[0].reshape(1,28,28))


# In[28]:


model.predict(X_test[0].reshape(1,28,28)).argmax(axis = 1)


# In[29]:


plt.imshow(X_test[1])


# In[30]:


model.predict(X_test[1].reshape(1,28,28)).argmax(axis = 1)


# In[31]:


model.predict(X_test[2].reshape(1,28,28)).argmax(axis = 1)


# In[32]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])


# In[33]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

