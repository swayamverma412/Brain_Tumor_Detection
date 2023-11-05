#!/usr/bin/env python
# coding: utf-8

# ### Import module

# In[1]:


#Import the necessary libraries first
import tensorflow as tf
import os
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.utils import load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array
from tensorflow.keras.metrics import binary_crossentropy
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D,GlobalAveragePooling2D
from keras.layers import Activation, Dropout, BatchNormalization, Flatten, Dense, AvgPool2D,MaxPool2D
from keras.models import Sequential, Model
from keras import backend as K
import shutil
import glob
import re


# In[2]:


data = '.brain-mri-images-for-brain-tumor-detection/'
No_brain_tumor = 'no/'
Yes_brain_tumor = 'yes/'


# ### Creating a dataframe showing tumour class with corresponding filepath

# In[3]:


dirlist=[No_brain_tumor, Yes_brain_tumor]
# ['brain_tumor_dataset/no/', 'brain_tumor_dataset/yes/']
classes=['No', 'Yes']
filepaths=[]
labels=[]
for i,j in zip(dirlist, classes):


    filelist=os.listdir(i)
    print(filelist)
    print('\n')
# os.listdir --> returns a list containing the names of the entries in the directory given by path.
    for f in filelist:
        filepath=os.path.join (i,f)
# os.path.join('brain_tumor_dataset/no/','1 no.jpeg;)
# brain_tumor_dataset/no/1 no.jpeg
        filepaths.append(filepath)
# store the path into empty list called filepaths
        labels.append(j)
    print(filepaths)
    print('\n')
    print(labels)
    print('\n')
print ('filepaths: ', len(filepaths), '   labels: ', len(labels))


# In[4]:


Files=pd.Series(filepaths, name='filepaths')
Label=pd.Series(labels, name='labels')
df=pd.concat([Files,Label], axis=1)
# df=pd.DataFrame(np.array(df).reshape(253,2), columns = ['filepaths', 'labels'])
# df.head()
df


# In[6]:


print(len(df))
print(len(df['labels']))



# In[7]:


df_dummies = pd.get_dummies(df['labels'], prefix='label')


# In[8]:


df["labels"].loc[df["labels"]=="Yes"]=1.0
df["labels"].loc[df["labels"]=="No"]=0.0


# In[ ]:





# ### Visualize the image of brain tumour

# In[9]:


plt.figure(figsize=(4,4))
for i in range(0,10):
    fig, ax = plt.subplots(figsize=(4,4))
    img = mpimg.imread(df['filepaths'][i])
    img_name = re.sub(r'^\D+','',df['filepaths'][i])
    ax.imshow(img)
    ax.set_title(img_name)


# Since each image has different size, we need to resize them into same size

# In[10]:


from PIL import Image
widths = []
heights = []
for idx, row in df.iterrows():
    path = row['filepaths']
#   print(path)
# brain_tumor_dataset/no/1 no.jpeg
# brain_tumor_dataset/no/10 no.jpg
# brain_tumor_dataset/no/11 no.jpg
# brain_tumor_dataset/no/12 no.jpg
# brain_tumor_dataset/no/13 no.jpg
# brain_tumor_dataset/no/14 no.jpg
    im = Image.open(path)
    
#     print(im)
# <PIL.JpegImagePlugin.JpegImageFile image mode=L size=630x630 at 0x1888E54F7C0>
# <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=173x201 at 0x1888E59BFA0>
# <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=300x168 at 0x1888E540340>
# <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=275x183 at 0x1888D29BFA0>
# <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=300x168 at 0x1888E540C70>
# <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=177x197 at 0x1888E5317F0>
# <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=232x217 at 0x1888E540C70>
#     print(im.size)
# (630, 630)
# (173, 201)
# (300, 168)
# (275, 183)
# (300, 168)
    width, height = im.size
    widths.append(width)
    heights.append(height)
avg_width = int(sum(widths) / len(widths))
avg_height = int(sum(heights) / len(heights))
print(avg_width, avg_height)


# The average width is 354 and the average height is 386.
# Making the width and height equal makes it simpler to do transformations later. So we’ll resize images to 300x300.

# In[11]:



# Image Resize Function
def load_resize_color_image(path):
    # load image and resize to 300x300
    image = load_img(path,target_size=(300,300))
    return image


# In[12]:


load_img('no/1_no.jpeg')


# In[33]:


# Load the image
img = Image.open('no/1_no.jpeg')

# Resize the image to the target size (300x300)
img = img.resize((300, 300))

# Save the resized image to another file
img.save('path_to_save/1_no_resized.jpeg')


# In[14]:


image_list =[]
cancer_list =[]
for idx, row in df.iterrows():
    path = row['filepaths']
    cancer = row['labels']
    image = load_resize_color_image(path)
    # turn image to array
    image_array = img_to_array(image)   
    image_list.append(image_array)
    cancer_list.append(cancer)


# In[15]:


print(image_list[0:5])


# In[16]:


print(cancer_list[0:5])


# In[30]:


from PIL import Image
widths = []
heights = []
for idx, row in df.iterrows():
    path = row['filepaths']
#   print(path)
# brain_tumor_dataset/no/1 no.jpeg
# brain_tumor_dataset/no/10 no.jpg
# brain_tumor_dataset/no/11 no.jpg
# brain_tumor_dataset/no/12 no.jpg
# brain_tumor_dataset/no/13 no.jpg
# brain_tumor_dataset/no/14 no.jpg
    im = Image.open(path)
    
#     print(im)
# <PIL.JpegImagePlugin.JpegImageFile image mode=L size=630x630 at 0x1888E54F7C0>
# <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=173x201 at 0x1888E59BFA0>
# <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=300x168 at 0x1888E540340>
# <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=275x183 at 0x1888D29BFA0>
# <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=300x168 at 0x1888E540C70>
# <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=177x197 at 0x1888E5317F0>
# <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=232x217 at 0x1888E540C70>
#     print(im.size)
# (630, 630)
# (173, 201)
# (300, 168)
# (275, 183)
# (300, 168)
    width, height = im.size
    widths.append(width)
    heights.append(height)
avg_width = int(sum(widths) / len(widths))
avg_height = int(sum(heights) / len(heights))
print(avg_width, avg_height)


# ### Shuffle the image and label

# In[17]:


from sklearn.utils import shuffle
image_list, cancer_list = shuffle(image_list, cancer_list)


# ### Define our X, y for train-test-split

# In[18]:


X_data = np.array(image_list)
y_data = np.array(cancer_list)


# In[19]:


print(X_data.shape)
print(y_data.shape)


# ### Split into training set and testing set

# In[20]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, train_size=0.8, random_state=0)


# In[21]:


print(X_train.shape)
print(y_train.shape)


# In[22]:


X_train = X_train/255
X_test = X_test/255


# In[23]:


print(X_test.shape)
print(y_test.shape)


# ### Building CNN Model Architecture

# In[24]:


input_shape = (300,300,3)


# In[26]:


model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
# ANN structure
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))


# In[27]:


model.summary()


# In[28]:


#   define compile to minimize categorical loss, use ada delta optimized, and optimize to maximizing accuracy
model.compile(loss="binary_crossentropy", optimizer='Adam', metrics=['accuracy'])

#   Train the model and test/validate the mode with the test data after each cycle (epoch) through the training data
#   Return history of loss and accuracy for each epoch
hist = model.fit(X_train, y_train,
          batch_size=10,
          epochs=10,
          verbose=1,
          validation_data=(X_test, y_test))


# In[29]:


score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#   Plot data to see relationships in training and validation data
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
epoch_list = list(range(1, len(hist.history['accuracy']) + 1))  # values for x axis [1, 2, ..., # of epochs]
plt.plot(epoch_list, hist.history['accuracy'], epoch_list, hist.history['val_accuracy'])
plt.legend(('Training Accuracy', 'Validation Accuracy'))
plt.show()


# In[ ]:


model.save('brain_tumor_model.h5')


# In[ ]:




