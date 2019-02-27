#!/usr/bin/env python
# coding: utf-8

# ### Analyzing the Stroop Effect
# Perform the analysis in the space below. Remember to follow [the instructions](https://docs.google.com/document/d/1-OkpZLjG_kX9J6LIQ5IltsqMzVWjh36QpnP2RYpVdPU/pub?embedded=True) and review the [project rubric](https://review.udacity.com/#!/rubrics/71/view) before submitting. Once you've completed the analysis and write-up, download this file as a PDF or HTML file, upload that PDF/HTML into the workspace here (click on the orange Jupyter icon in the upper left then Upload), then use the Submit Project button at the bottom of this page. This will create a zip file containing both this .ipynb doc and the PDF/HTML doc that will be submitted for your project.
# 
# 
# (1) What is the independent variable? What is the ?

# independent variable:  congruency  
# 
# dependent variable : the time spended 

# (2) What is an appropriate set of hypotheses for this task? Specify your null and alternative hypotheses, and clearly define any notation used. Justify your choices.

# 1- my hypotheses is :
# null hypotheses : congruency does not affect the time to complet the task 
# 
# alternative hypotheses : congruency affect the time to complete the task 
# 
# 2- what i expect is that i perform the linear regression. becuase the dependent variable is continuous not categorical
# 

# (3) Report some descriptive statistics regarding this dataset. Include at least one measure of central tendency and at least one measure of variability. The name of the data file is 'stroopdata.csv'.

# In[44]:


# Perform the analysis here
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm

df = pd.read_csv('stroopdata.csv')
df.head()

measure_Congruent = df['Congruent'].mean()
measure_Incongruent = df['Incongruent'].mean()



measure_Congruent ,measure_Incongruent


# In[38]:



standard_Congruent = df['Congruent'].std()
standard_Incongruent = df['Incongruent'].std()


standard_Congruent ,standard_Incongruent 


# (4) Provide one or two visualizations that show the distribution of the sample data. Write one or two sentences noting what you observe about the plot or plots.

# In[39]:



plt.hist(df['Congruent'] )


# the plpot shows that the largest group of people is 6 who took 15 seconds to complet the task , the data i snormaly  distributed

# In[40]:


plt.hist(df['Incongruent'] )


# the plot shows the largest group of people 6 took between 20 to 22 seconds to complet the task , there is outliers 2 people who took between 33 to 35 secinds 

# (5)  Now, perform the statistical test and report your results. What is your confidence level or Type I error associated with your test? What is your conclusion regarding the hypotheses you set up? Did the results match up with your expectations? **Hint:**  Think about what is being measured on each individual, and what statistic best captures how an individual reacts in each environment.

# 1- my confidence level is 90 , 5 from both sides 
# thou my observation size is 24, referring to a statistical table I deduct a crtical statistic t-value of  1.318.
# my independent variable`s t-value is 6.532, it`s greater than 1.318, so i can reject my hypotheses 
# 
# my resul maches with my expectations, this test confuse the human`s brin 
# 
# 

# In[41]:



test = np.repeat('Congruent',24)
time = df['Congruent']


df2 = pd.DataFrame({'Time':time,'Test':test})
df2.head()


time = df['Incongruent']
Test = np.repeat('Incongruent',24)

#create a dataframe with the incongruent values
df3 = pd.DataFrame({'Time':time,'Test':Test})
df3.head()


df4 = df2.append(df3,ignore_index=True)
df4.reset_index(drop=True)
df4.head()


# In[42]:



df4[['Congruent','Incongruent']]= pd.get_dummies(df4['Test'])
df4 = df4.drop('Congruent',axis = 1)

df4.head()


# In[45]:


df4 = df4.rename(columns={'Incongruent': 'Variable'})
stats.chisqprob = lambda chisq, new_df: stats.chi2.sf(chisq, df4)

df4['intercept'] = 1

lm = sm.OLS(df4['Time'],df4[['intercept','Variable']])
results = lm.fit()
results.summary()


# In[ ]:





# (6) Optional: What do you think is responsible for the effects observed? Can you think of an alternative or similar task that would result in a similar effect? Some research about the problem will be helpful for thinking about these two questions!

# --write answer here--
