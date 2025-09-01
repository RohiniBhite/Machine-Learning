#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import scikit-learn dataset library
from sklearn import datasets

#Load dataset
cancer = datasets.load_breast_cancer()


# In[3]:


# print the names of the 13 features
print("Features: ", cancer.feature_names)


# In[4]:


# print the label type of cancer('malignant' 'benign')
print("Labels: ", cancer.target_names)


# In[5]:


# print data(feature)shape
cancer.data.shape


# In[6]:


# print the cancer data features (top 5 records)
print(cancer.data[0:5])


# In[7]:


# print the cancer labels (0:malignant, 1:benign)
print(cancer.target)


# In[8]:


# Import train_test_split function
from sklearn.model_selection import train_test_split

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.3,random_state=109) # 70% training and 30% test


# In[19]:


#Import svm model
from sklearn import svm

#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)


# In[16]:


from sklearn import metrics
print(metrics.confusion_matrix(y_test,y_pred))


# In[20]:


labels = [0,1]
cm = confusion_matrix(y_test, y_pred, labels=labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot();


# In[14]:


#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
print("F1_Score:",metrics.f1_score(y_test, y_pred))


# In[ ]:




