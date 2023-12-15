#!/usr/bin/env python
# coding: utf-8

# **GUID:** 2507608
# 
# **GitHub URL:** (https://github.com/lordofanywhere/ai-for-the-arts-a/)[https://github.com/lordofanywhere/ai-for-the-arts-a/]

# # Week 1: Getting started with Anaconda, Jupyter Notebook and Python
# 
# Exercises o familiarise myself with Jupyter Notebook and its relationship with Python
# 
# __a)	Why you chose to join this course – for, motivation, vision, aspiration?__
#         I chose this course because I am very excited about the progress in AI technologies andc its potential for change the way we create and curate art.
#         
# __b)	Prior experience, if any, you have with AI and/or Python__ 
#         No, this is the first time I use Python. I have used generative AI tools, however.
#         
# __c)	What you expect to learn from the course (aim for 3-5 bullet points)__
#         - The different types of AI and how they can be used to create and curate art
#         - Use AI tools practically tocreate art
#         - Technical, social and cultural dimensions of the use of AI

# In[ ]:


print ("Hello World!")


# In[ ]:


message = "Hello World!"
print (message)


# In[ ]:


message = "Hello World, my name is Emilio! This is my first incursion into AI"
print (message)


# In[ ]:


message = "Hello World, my name is Emilio! This is my first incursion into AI"
print (message + message)


# In[ ]:


message = "Hello World, my name is Emilio! This is my first incursion into AI"
print (message*3)


# In[ ]:


message = "Hello World, my name is Emilio! This is my first incursion into AI"
print (message[0])


# In[ ]:


message = "Hello World, my name is Emilio! This is my first incursion into AI"
print (message[2])


# In[ ]:


message = "Hello World, my name is Emilio! This is my first incursion into AI"
print (message[1])


# In[ ]:


greeting = "Hello World!"
print (greeting)


# In[ ]:


print (message + greeting)


# In[ ]:


from IPython.display import *


# In[ ]:


YouTubeVideo("4eA-X4P5EBo")


# In[ ]:


import webbrowser
import requests

print("Shall we hunt down an old website?")
site = input("Type a website URL: ")
era = input("Type year, month, and date, e.g., 20150613: ")
url = "http://archive.org/wayback/available?url=%s&timestamp=%s" % (site, era)
response = requests.get(url)
data = response.json()
try:
    old_site = data["archived_snapshots"]["closest"]["url"]
    print("Found this copy: ", old_site)
    print("It should appear in your browser.")
    webbrowser.open(old_site)
except:
    print("Sorry, could not find the site.")
# anotating comments


# # Week 2: Exploring Data in Multiple Ways

# In[ ]:


from IPython.display import Image


# In[ ]:


Image ("picture1.jpg")


# In[ ]:


Image ("picture1.jpg")


# In[ ]:


Audio ("audio1.mid")


# In[ ]:


Audio ("audio2.ogg")


# The OGG file plays but the MID file does not play, probably due to the codecs and libraries available on the local computer used.

# In[ ]:


# This file is licensed under the Creative Commons Attribution-Share Alike 3.0 Unported license.
# You are free: 
# •	to share – to copy, distribute and transmit the work
# •	to remix – to adapt the work
# Under the following conditions: 
# •	attribution – You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
# •	share alike – If you remix, transform, or build upon the material, you must distribute your contributions under the same or compatible license as the original.
# The original ogg file was found at the url: 
# https://en.wikipedia.org/wiki/File:GoldbergVariations_MehmetOkonsar-1of3_Var1to10.ogg


# In[ ]:


from matplotlib import pyplot


# In[ ]:


from matplotlib import pyplot
test_picture = pyplot.imread("picture1.jpg")
print("Numpy array of the image is: ", test_picture)
pyplot.imshow(test_picture)


# In[ ]:


test_picture_filtered = 2*test_picture/3


# In[ ]:


pyplot.imshow(test_picture_filtered)


# In this plot chart, I believe we can see the image, with the CMY (cyan–magenta–yellow) colour points filtered. 

# # Week 3: Exploring scikit-learn (a.k.a sklearn)

# In[ ]:


from sklearn import datasets


# In[ ]:


dir(datasets)


# I have chosen 'load_breast_cancer' and 'load_diabetes' as I am curious as to what these mean in this context.

# In[ ]:


breast_cancer_data = datasets.load_breast_cancer()


# In[ ]:


diabetes_data = datasets.load_diabetes()


# In[ ]:


breast_cancer_data.DESCR


# In[ ]:


print(breast_cancer_data.DESCR)

diabetes_data.DESCR
# In[ ]:


print(diabetes_data.DESCR)


# In[ ]:


breast_cancer_data.feature_names


# In[ ]:


diabetes_data.feature_names


# In[ ]:


breast_cancer_data.target_names


# In[ ]:


diabetes_data.keys()


# In[ ]:


diabetes_data.target


# In[ ]:


from sklearn import datasets
import pandas

breast_cancer_data = datasets.load_breast_cancer()

breast_cancer_dataframe = pandas.DataFrame(data=breast_cancer_data['data'], columns = breast_cancer_data['feature_names'])


# In[ ]:


breast_cancer_dataframe.head()


# In[ ]:


breast_cancer_dataframe.describe


# In[ ]:


wine_data = datasets.load_wine()


# In[ ]:


wine_data.DESCR


# In[ ]:


print(wine_data.DESCR)


# In[ ]:


wine_data.feature_names


# In[ ]:


wine_data.keys()


# In[ ]:


from sklearn import datasets
import pandas

wine_data = datasets.load_wine()

wine_dataframe = pandas.DataFrame(data=wine_data['data'], columns = wine_data['feature_names'])


# In[ ]:


wine_dataframe.head()


# In[ ]:


wine_dataframe.describe()


# ## Discussion: Basic Data Exploration with Python library Pandas
# 
# I believe that the command wine_dataframe.head() creates a table for the elements on the database, and apply headers based on the attributes of the data stored.
# 
# The command wine_dataframe.describe() describes some statistics of the dataset.

# ## Discussion: Thinking about data bias

# First I would explore and understand how the data in the datasets I'm using was generated. Once I understand the potential biases, I would create a strategy to pre-process the data or obtain additional data, and an Exploratory Data Analysis strategy.
# 
# I would consider using datasets and creating an open model, so I can make it publicly available for data scrutiny.
# 
# I would balance any data procedent from reviews or social media posts with data obtained from a representative sample of people that would not have shared their opinion unprompted (through a survey, for example).
# 
# Another similar approach would be balancing responses to popular items (e.g. in recommendation systems) with prompting responses for items without a known rating, as well as randomising the way these are presented, or using propensity weighting techniques.
# 
# I would also seek to balance the data collected for the model by collecting data from demographics not represented in the initial datasets.
