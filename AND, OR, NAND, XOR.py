#!/usr/bin/env python
# coding: utf-8

# **Experiment 2**
# 
# Name: Komal Chitnis
# 
# Class: BE-A
# 
# Moodle id: 20102068
# 

# In[2]:


# Neural Network for AND function
import numpy as np
class and_NN:
  def __init__(self,iv):
    self.input_vector = np.array(iv)
    self.weight_vector = np.array([[-1.5,1,1]])
    self.output = np.array([])

  def compute(self):
    mfv = np.c_[np.ones(4),self.input_vector] # padding with bias
    result = mfv.dot(self.weight_vector.transpose()) # multipy
    self.output = [0 if r < 0 else 1 for r in result] # simple filter

  def show_result(self):
    print("Input Vector : \n",self.input_vector)
    print("Output : \n",self.output)

  def get_output(self):
    return self.output

n = and_NN(np.array([[0,0],[0,1],[1,0],[1,1]]))
n.compute()
n.show_result()


# In[3]:


# Neural Network for OR function
import numpy as np

class or_NN:
  def __init__(self,iv):
    self.input_vector = np.array(iv)
    self.weight_vector = np.array([[-0.5,1,1]])
    self.output = np.array([])

  def compute(self):
    mfv = np.c_[np.ones(4),self.input_vector] # padding with bias
    result = mfv.dot(self.weight_vector.transpose()) # multipy
    self.output = [0 if r < 0 else 1 for r in result] # simple filter

  def show_result(self):
    print("Input Vector : \n",self.input_vector)
    print("Output : \n",self.output)

  def get_output(self):
    return self.output


n = or_NN(np.array([[0,0],[0,1],[1,0],[1,1]]))

n.compute()
n.show_result()


# In[5]:


# Neural Network for NAND function
import numpy as np

class nand_NN:
  def __init__(self,iv):
    self.input_vector = np.array(iv)
    self.weight_vector = np.array([[1.5,-1,-1]])
    self.output = np.array([])

  def compute(self):
    mfv = np.c_[np.ones(4),self.input_vector] # padding with bias
    result = mfv.dot(self.weight_vector.transpose()) # multipy
    self.output = [0 if r < 0 else 1 for r in result] # simple filter
  def show_result(self):
    print("Input Vector : \n",self.input_vector)
    print("Output : \n",self.output)

  def get_output(self):
    return self.output


n = nand_NN(np.array([[0,0],[0,1],[1,0],[1,1]]))

n.compute()
n.show_result()


# In[7]:


# Neural Network for XOR function
# NN for XOR is realized using OR, NAND and AND functions
# a xor b = (a or b) and (a nand b)
# or and nand in the first layer, and in the second layer

class xor_NN:
  def __init__(self,iv):
    self.input_vector = np.array(iv) # Input Layer
    self.output = np.array([])

  def compute(self):
    or_neuron = or_NN(self.input_vector) # Layer 1
    nand_neuron = nand_NN(self.input_vector) # Layer 1
    or_neuron.compute()
    nand_neuron.compute()
    h1 = or_neuron.get_output()
    h2 = nand_neuron.get_output()
    h = np.concatenate((h1,h2))
    iv_l2= h.reshape(4,2)
    and_neuron = and_NN(iv_l2) # Layer 2
    and_neuron.compute()
    self.output = and_neuron.get_output()

  def show_result(self):
    print("Input Vector : \n",self.input_vector)
    print("Output : \n",self.output)

  def get_output(self):
    return self.output

n = xor_NN(np.array([[0,0],[0,1],[1,0],[1,1]]))

n.compute()
n.show_result()

