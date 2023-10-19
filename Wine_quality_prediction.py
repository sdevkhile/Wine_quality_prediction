#!/usr/bin/env python
# coding: utf-8

##### Importing the necessary libraries

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')

from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split , GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, auc,confusion_matrix,classification_report,mean_squared_error
from sklearn.metrics import roc_curve,plot_confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


# In[5]:


#  Gather the data

df = pd.read_csv('QualityPrediction.csv')


# In[6]:


df.head()


# In[7]:


#  Our target variable is the quality and rest of the variables are predictor varibales. 
# The target variable is multi-categorical in nature and falls under ordinal datatype. 

df.tail()


# In[182]:


df.describe()
#  Out all the features, free sulfur dioxide and total sulfur dioxide is deviated/dispersed compared to other features
#  due to high standard deviation.


# In[143]:


df.shape 
#There are in total 1599 rows and 12 columns.


# In[9]:


df.info()
# There are no null values in the dataset. The column 'quality' is our target variable and remaining columns
#  are predictor variables. 


# In[10]:


df['quality'].value_counts()*100/len(df['quality'])


# In[11]:


# Majority of the wines have quality in the range of 5,6 and 7.  
df.quality.value_counts()


# In[12]:


df.dtypes
# There are 11 continuous features and 1 categorical variable which is our Target variable. 


# In[13]:


# Detecting Outliers:

def z_outliers(data):
    outlier= []
    mean = np.mean(data)
    std = np.std(data)
    for i in data:
        z = (i-mean)/std
        if z > 3 or z < -3:
            outlier.append(i)
    print('Outliers in the dataset are:' , outlier)  
z_outliers(df['citric acid']) 


# In[14]:


z_outliers(df['alcohol']) 


# In[15]:


z_outliers(df['chlorides']) 


# In[183]:


z_outliers(df['free sulfur dioxide']) 


# In[184]:


z_outliers(df['total sulfur dioxide']) 


# In[16]:


#  Assigning the target variable as 'y' and predictor variables as 'x'.  
x = df.drop('quality', axis = 1).values


# In[17]:


x


# In[18]:


x.shape


# In[19]:


y = df.quality.values.reshape(-1,1)


# In[20]:


y.shape


# In[21]:


#  Data Visualization


# In[22]:


#  Let's try to understand the data better with the help of Data Visualization.


# In[156]:


#  From the below lineplots we see how our Target variable 'quality' changes with respect to predictor variables. 

fig , axes = plt.subplots(2,2, figsize = (13,11))
sns.lineplot(df.quality, df.sulphates, ax = axes[0,0])
axes[0,0].set_xlabel('Quality--->')
axes[0,0].set_ylabel('Sulphates--->')
axes[0,0].set_title('Plot 1')

sns.lineplot(df['quality'], df['volatile acidity'] , ax = axes[0,1])
axes[0,1].set_xlabel('Quality--->')
axes[0,1].set_ylabel('Volatile Acidity--->')
axes[0,1].set_title('Plot 2')

sns.lineplot(df['quality'], df['citric acid'], ax = axes[1,0])
axes[1,0].set_xlabel('Quality--->')
axes[1,0].set_ylabel('Citric Acid--->')
axes[1,0].set_title('Plot 3')

sns.lineplot(df.quality, df.chlorides, ax = axes[1,1])
axes[1,1].set_xlabel('Quality--->')
axes[1,1].set_ylabel('Chlorides--->')
axes[1,1].set_title('Plot 4')


# In[24]:


plt.figure(figsize=(12,8))
sns.histplot(data=df , x = 'total sulfur dioxide', hue = 'quality', palette=['red', 'green', 'blue', 'orange', 'yellow', 'black'])

# From the below plot we see there are only 2 wines of quality 7 that have total sulfur dioxide more than 260 which is far 
#  beyond the normal range of total sulfur dioxide of ~ (0 to 170)


# In[25]:


plt.figure(figsize=(12,8))
sns.histplot(data=df , x = 'alcohol', hue = 'quality', palette=['red', 'green', 'blue', 'orange', 'purple', 'black'])


# In[26]:


# To check how the independent variables are related to each other we will calculate Multicollinearity.


# In[27]:


df.corr()


# In[28]:


plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), cmap = 'YlGnBu', annot = True)
plt.show()


# In[29]:


list(df.columns)


# In[130]:


X_new = df[['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide',
'total sulfur dioxide','density','pH','sulphates','alcohol']]   # X_new only has predictor variables
sns.pairplot(data= X_new)
# From the below pairplots we see how each predictor variable is related to the other predictor variable. This helps
# us to analyse how some of the input variables are dependent on each other.


# In[31]:


vif_data = pd.DataFrame()


# In[32]:


vif_data['features'] = X_new.columns


# In[33]:


vif_data


# In[34]:


vif_data['VIF'] = [variance_inflation_factor(X_new,i) for i in range(11)]


# In[35]:


vif_data


# In[36]:


#  Predictor variables are highly related to each other. If the VIF is greater than 5 it means there is Multicollinearity
#   between the predictor variables. Hence we will not leverage Linear Regression and Logistic Regression models as the
#  independent variables are having high correlation amongst each other.


# In[37]:


dupe = df.duplicated()


# In[38]:


dupe


# In[39]:


#  There are about 240 duplicate rows in the dataframe. 
#  In this case we will have to investigate and discuss further with the client if the duplicates are legit or 
#  needs to be discarded.  

len(dupe)


# In[40]:


dupe


# In[41]:


# Splitting the data into train and test datasets


# In[42]:


xtrain , xtest, ytrain, ytest =  train_test_split(x,y,test_size=0.2,random_state=5)


# In[43]:


#  Scaling is always performed after splitting the data into train and test in order to avoid the test data getting exposed 
#  to the model during Model building stage which will cause Data leakage. However we will not use scaling here
#  since the data is not distance based. 


# In[44]:


xtrain


# In[45]:


xtrain.shape


# In[46]:


xtest


# In[47]:


xtest.shape


# In[48]:


# Now that we have performed EDA on the data to understand it better, let us explore different Machine Learning algorithms on 
# train and test datasets and see how each model performs:


# ### Decision Tree 

# In[49]:


# We will create decision tree using both the methods i.e. GINI and Entropy.


# In[50]:


#  Decision Tree with depth 4 (GINI):


# In[51]:


model_dt_4 = DecisionTreeClassifier(random_state=4, max_depth =4)


# In[52]:


# Model creation:

model_dt_4.fit(xtrain,ytrain)


# In[53]:


y_pred_4 = model_dt_4.predict(xtest) # Testing the model on unseen data i.e. Test data
accuracy_score_4 = accuracy_score(ytest,y_pred_4)
print('Accuracy Score for model with depth 4 is: ',accuracy_score_4)


# In[54]:


#  Decision Tree with depth 6 (GINI):


# In[55]:


model_dt_6 = DecisionTreeClassifier(random_state=6,max_depth=6)


# In[56]:


model_dt_6.fit(xtrain,ytrain)


# In[57]:


y_pred_6 = model_dt_6.predict(xtest) 
accuracy_score_6 = accuracy_score(ytest,y_pred_6)
print('Accuracy Score for model with depth 6 is: ',accuracy_score_6)


# In[58]:


#  Decision Tree with depth 8 (GINI):


# In[59]:


model_dt_8 = DecisionTreeClassifier(random_state=8,max_depth=8)


# In[60]:


model_dt_8.fit(xtrain,ytrain)


# In[61]:


y_pred_8 = model_dt_8.predict(xtest) 
accuracy_score_8 = accuracy_score(ytest,y_pred_8)
print('Accuracy Score for model with depth 8 is: ',accuracy_score_8)


# In[62]:


# Decision Tree using Entropy


# In[63]:


model_dt_ent = DecisionTreeClassifier(random_state=8,max_depth=8, criterion='entropy')


# In[64]:


model_dt_ent.fit(xtrain,ytrain)


# In[65]:


y_pred_ent = model_dt_ent.predict(xtest) 
accuracy_score_ent = accuracy_score(ytest,y_pred_ent)
print('Accuracy Score for model with depth 8 using Entropy is: ',accuracy_score_ent)


# In[66]:


classificationReport_dt = classification_report(ytest,y_pred_8)


# In[67]:


print(classificationReport_dt)


# In[163]:


#  To check overfitting/underfitting:

y_pred_dt8_train = model_dt_8.predict(xtrain)


# In[164]:


accuracy_score(ytrain,y_pred_dt8_train)


# In[165]:


#  The model seems to be neither underfitting not overfitting in this case as there is not much significant difference between
# the prediction obtained from trained data (78%) and test data (63%).


# ### Random Forest 

# In[68]:


model_rf = RandomForestClassifier()


# In[69]:


#  We will hypertune the parameters with the help of GridSearchCV which will provide us with correct set of 
#  parameters that we can leverage for training the model.


# In[70]:


param_dist = {'max_depth' : [2,4,6,8], 'criterion' : ['gini', 'entropy'], 'bootstrap' : [True, False], 
              'max_features' : ['auto', 'sqrt', 'log2', None]}


# In[71]:


cv_rf = GridSearchCV(model_rf, cv =10, param_grid = param_dist, verbose = 1 , n_jobs = 3) # Run the GridSearchCV to achieve all
#  possible PnCs of these parameters


# In[72]:


cv_rf.fit(xtrain,ytrain)


# In[73]:


print('Best parameters using GridSearchCV are: \n', cv_rf.best_params_)


# In[74]:


model_rf.set_params(criterion = 'entropy', max_depth = 8, max_features = 'auto', bootstrap = False )


# In[75]:


model_rf.fit(xtrain,ytrain)
y_pred_rf = model_rf.predict(xtest)


# In[76]:


accuracy_score(ytest,y_pred_rf)


# In[150]:


plot_confusion_matrix(model_rf,xtest,ytest)


# In[77]:


classificationReport_rf =  classification_report(ytest, y_pred_rf)
print(classificationReport_rf)


# In[78]:


print(np.squeeze(ytest))


# In[79]:


print(y_pred_rf)


# In[159]:


#  To check overfitting/underfitting:

y_pred_train_rf =  model_rf.predict(xtrain)


# In[161]:


accuracy_score(ytrain,y_pred_train_rf)


# In[162]:


print(classification_report(ytrain,y_pred_train_rf))


# In[ ]:


#  The model seems to be overfitting as it is performing better on trained dataset. It achieved good accuracy score of 
# 91% on trained data whereas on the test data (unseen data) the model achieved only 71%. Also we see from above 
#  classification report the model is performing very well in terms of other evalaution parameters as well. 


# ### Gaussian Naive Bayes

# In[80]:


model_gnb = GaussianNB()

# Since the predictor variables are highly related to each other due to high Multicollinearity, the assumption of this algorithm
# that predictor variables are not related to each other at all does not fall in line in this dataset. This will affect  
# the accuracy of the model.


# In[81]:


model_gnb.fit(xtrain,ytrain)


# In[82]:


y_pred_gnb = model_gnb.predict(xtest)


# In[83]:


# Accuracy obtained from this model is below average. 
accuracy_score_gnb = accuracy_score(ytest,y_pred_gnb)
print(accuracy_score_gnb)                       


# In[84]:


plot_confusion_matrix(model_gnb,xtest,ytest)


# In[85]:


classification_report_gnb = classification_report(ytest,y_pred_gnb)
print(classification_report_gnb)

# From the below classification report we see that accuracy is not very good. It is due to the fact of Multicollinearity 
# is high which contradicts the assumption of this algorithm. 


# In[ ]:


#  To check overfitting/underfitting:


# In[169]:


y_pred_train_gnb = model_gnb.predict(xtrain)


# In[170]:


accuracy_score(ytrain,y_pred_train_gnb)


# In[ ]:


#  The model seems to be underfitting as the performance obtained from both trained and test dataset is bad.


# ###  K Nearest Neighbours with Cross Validation

# In[86]:


# KNN is a distance based model hence we will first scale our features before training the model.

ss = StandardScaler()


# In[87]:


xtrain_ss = ss.fit_transform(xtrain)
xtest_ss = ss.transform(xtest)


# In[88]:


from sklearn.neighbors import KNeighborsClassifier


# In[89]:


# Taking the K value as 3 at first and observing how the model performs. 

knn = KNeighborsClassifier(n_neighbors = 3)


# In[90]:


knn.fit(xtrain_ss,ytrain)


# In[91]:


y_pred_knn3 = knn.predict(xtest_ss)


# In[132]:


classification_report_knn = classification_report(ytest,y_pred_knn3)
print(classification_report_knn)


# In[93]:


#  Choosing K value:-
#  We will apply loop for values of K from 1 to 45 and decide which values of K will help us achieve good accuracy rate. 
#  And also we need to make sure our model is stable which can be checked by plotting the graph between accuracy rate/error rate
#  and the respective K values.


# In[94]:


accuracy_rate = []
for i in range(1,45):
    knn = KNeighborsClassifier(n_neighbors = i)
    score= cross_val_score(knn,x,y,cv=10)
    accuracy_rate.append(score.mean())


# In[95]:


error_rate = []
for i in range(1,45):
    knn = KNeighborsClassifier(n_neighbors = i)
    score = cross_val_score(knn,x,y,cv=10)
    error_rate.append(1-score.mean())
    


# In[96]:


plt.figure(figsize=(8,6))
plt.plot(range(1,45), accuracy_rate, color = 'blue', linestyle = 'dashed', marker='o', markersize=10, markerfacecolor='red')
plt.title('Accuracy rate vs. K value')
plt.xlabel('K Value')
plt.ylabel('Accuracy rate')


# In[97]:


plt.figure(figsize=(8,6))
plt.plot(range(1,45), error_rate , color = 'blue', linestyle = 'dashed', marker='o', markersize=10, markerfacecolor='red')
plt.title('Accuracy rate vs. K value')
plt.xlabel('K Value')
plt.ylabel('Accuracy rate')


# In[139]:


#  At around K values of 12 to 15 in the above graphs we see there is some stability however beyond these points there is  
#  steep rise/decline in the accuracy rate. Lets see how the model performs at K = 15. 
knn = KNeighborsClassifier(n_neighbors = 15 )


# In[134]:


knn.fit(xtrain_ss,ytrain)


# In[135]:


y_pred_knn15 = knn.predict(xtest_ss)


# In[137]:


print(classification_report(ytest,y_pred_knn15))


# In[172]:


#  The graph is condensed and getting constant for K values beyond 23. At K=23 the accuracy rate improved to 62 from 60(K=15). 
#  This might be due to the fact that the accuracy rate beyond this point is not fluctuating. 

knn = KNeighborsClassifier(n_neighbors = 23 )
knn.fit(xtrain_ss,ytrain)
y_pred_knn23 = knn.predict(xtest_ss)
print(classification_report(ytest,y_pred_knn23))


# In[168]:




knn = KNeighborsClassifier(n_neighbors = 25 )
knn.fit(xtrain_ss,ytrain)
y_pred_knn25 = knn.predict(xtest_ss)
print(classification_report(ytest,y_pred_knn25))


# In[ ]:


#  To check overfitting/underfitting:


# In[179]:


knn = KNeighborsClassifier(n_neighbors = 23 )
knn.fit(xtrain_ss,ytrain)
y_pred_train_knn23 = knn.predict(xtrain_ss)
accuracy_score(ytrain,y_pred_train_knn23)


# In[180]:


#  KNN model at K=23 is neither underfitting not overfitting as the performance of the model obtained from both train and 
# test datasets does not have much difference.


# ### Support Vector ML model

# In[104]:



svc = SVC(random_state=4)

# Running the SVM with default hypertuned parameters:-


# In[105]:


svc.fit(xtrain_ss, ytrain)


# In[106]:


y_pred_svc = svc.predict(xtest_ss)


# In[107]:


print(classification_report(ytest,y_pred_svc))


# In[108]:


svc.C


# In[109]:


svc.kernel


# In[110]:


svc.gamma


# In[ ]:


# Optimizing the parameters from GridSearchCV


# In[112]:


tuned_parameters = {'C': [0.1,1,10], 'gamma': [1,0.1,0.001], 'kernel': ['rbf', 'poly', 'linear']}


# In[113]:


model_svm = GridSearchCV(svc,tuned_parameters,cv=10, verbose=1,n_jobs=3, scoring='accuracy')


# In[114]:


model_svm.fit(xtrain_ss,ytrain)


# In[115]:


print(model_svm.best_params_)


# In[116]:


svc =  SVC(C=10,gamma=1,kernel='rbf')


# In[117]:


svc.fit(xtrain_ss,ytrain)


# In[118]:


y_pred_hsvc = svc.predict(xtest_ss)


# In[120]:


print(classification_report(ytest,y_pred_hsvc))


# In[166]:


#  To check overfitting/underfitting:

y_pred_train_svc = svc.predict(xtrain_ss)


# In[167]:


accuracy_score(ytrain,y_pred_train_svc)


# In[144]:


# The SVM performed better with default parameters as compared to the hypertuned parameters obtained after
# Optimiziation since the accuracy rate decreased from 67 to 65. The model is overfitting as we see performance of the model is 
# 99% on trained data and only 65% on the test data. 


# ### Conclusion

# In[158]:


# If we take accuracy rate as our evaluation parameter we see that Random Forest is performing better than 
# other models with accuracy rate of 71% however the model is overfitting. From all the models KNN model 
#  at K = 23 and Decision Tree is neither overfitting nor underfitting and also giving an accuracy rate of 62-63%. 


# In[ ]:




