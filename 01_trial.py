#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install --upgrade scikit-image')


# In[2]:


import efficientnet.keras as efn 

model = efn.EfficientNetB0(weights='imagenet')  # or weights='noisy-student'


# In[3]:


import tensorflow as tf
import tensorflow.keras

print(tf.__version__)
print(tensorflow.keras.__version__)

