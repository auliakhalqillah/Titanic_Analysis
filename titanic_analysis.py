# Objective : Predict the survival of the passengers aboard RMS Titanic.
# Source    : https://www.kaggle.com/c/titanic/data
# Date      : 2020, October 13th

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

df_train = pd.read_csv('titanic_train.csv')
df_test = pd.read_csv('titanic_test.csv')
df_train.merge(df_test)
df = df_train
# Head data
print('[1] Head data:')
print(df.head())

# Shape (size) data
print('\n[2] Shape/Size data:')
print(df.shape)
# The titanic data has 891 rows and 12 columns

# Info data
print('\n[3] Info data:')
print(df.info())

# Total missing value
print('\n[4] Total missing value:')
print('Each item:', df.isnull().sum())
print('Total :', df.isnull().sum().sum())
# By info data, we get total 866 missing value, there are in Age (177 data), Cabin (687 data) and Embarked (2 data).
# The Cabin has high missing value, so the Cabin is removed from dataframe. 

# Remove Cabin, PassengerId, and Ticket from dataframe
df = df.drop(['Cabin','PassengerId', 'Ticket'], axis=1)
print('\n[5] After remove Cabin, PassengerId and Ticket:')
print(df.info())

# The missing value in Age column is replaced by mean of Age
print('\n[6] Fill missing value by mean:')
df['Age'].fillna(df['Age'].mean(), inplace = True) 
print(df.info())

# Two values are missing in Embarked data. To fill these missing values, We assume and replace by high fequency of Embarked.
print('\n[7] Frequency from each Embarkation:')
print(df.groupby('Embarked').size())
# Embarked
# C    168
# Q     77
# S    644
# From these results, the S is high frequency of embarkation. Then, We replace the missing values by S emberkation
df['Embarked'].fillna('S', inplace = True)
print(df.info())

# Short describe
print('\n[8] Describe data:')
print(df.describe())

# Based on the main objective, We can generate a question that are there correlation between survived and other features?
correaltion = df.corr()
print('\n[9] Correlation for each pair feature:')
print(correaltion)
# In generally, the survived is inversely correlated with Pclass (Passanger Class) as a higher correlation from other features.
# It means the passangers who survived are in the fisrt class (Pclass = 1), where the first class indicates high socio-economic status.
# The Pclass also inversely correlated with Fare where the first class has high fare (expensive) and third class has low fare (cheap). 
# It means, the passengers who survived are domainted in first class. On the other hand, the survived is weak correlated to Age, 
# SibSp (Siblings/Spouses) and Parch (Parrents/Children).

# How much passenger who aboard in RMS Titanic by Gender?
print('\n[10] Passanger who aboard in RMS titanic by Gender')
print('Female:', df[df['Sex'] == 'female']['Sex'].count())
print('Male:', df[df['Sex'] == 'male']['Sex'].count())

# How much passenger who survived by overall?
print('\n[11] Passanger who survived and did not survived')
print('Survived:', df[df['Survived'] == 1]['Survived'].count())
print('No Survived:', df[df['Survived'] == 0]['Survived'].count())

ax = sns.countplot(x = 'Survived', data = df)
ax.set_title('Number of Survived and didn\'t Survived')

# How much passanger who survived by Gender?
print('\n[12] Passanger who survived by Gender')
print('Female :',df[(df['Survived'] == 1) & (df['Sex'] == 'female')]['Survived'].count())
print('Male :',df[(df['Survived'] == 1) & (df['Sex'] == 'male')]['Survived'].count())

# How much passanger who not survived by Gender?
print('\n[13] Passanger who did not survived by Gender')
print('Female :',df[(df['Survived'] == 0) & (df['Sex'] == 'female')]['Survived'].count())
print('Male :',df[(df['Survived'] == 0) & (df['Sex'] == 'male')]['Survived'].count())

sns.catplot(x='Survived', col='Sex', kind='count', data = df)

# How much passanger who survived by Pclass?
print('\n[14] Passanger who survived and did not survived by Pclass')
print(df.groupby(['Survived','Pclass']).size())

sns.catplot(x='Survived',col='Pclass',data=df,kind='count')

# RMS Titanic was boarded 891 passengers consisting of 314 female passengers and 577 male passengers. 
# Based on these data, about 342 passengers are survived and 549 passengers are not survived. By 342 
# passengers are survived, as many as 233 passngers are female and 109 passengers are male. By 549 
# passengers are not survived, as many as 81 passengers are female and 468 passengers are male. 
# Female passengers are high number of survived than male passengers. Furthermore, the high number 
# of passangers who survived are in the first class about 136 passengers and the high number of 
# passengers who didn't survived about 372 passengers are in third class.

# How about Age distribution in RMS Titanic?
print('\n[15] Age distribution in RMS Titanic')
plt.figure(4, figsize=(10,6))
plt.hist(df['Age'], bins=10, edgecolor='black',label='Age distribution in RMS Titanic',alpha=0.5)
plt.hist(df[df['Survived'] == 1]['Age'], bins=10, edgecolor='black',label='Age distribution of survived',alpha=0.5)
plt.hist(df[df['Survived'] == 0]['Age'], bins=10, edgecolor='black',label='Age distribution of not survived',alpha=0.5)
plt.xlabel('Age')
plt.ylabel('count')
plt.title('Age Distributions')
plt.legend()
# Based on Age distributions, the passengers who aboard in RMS Titanic are dominated by passengers who has age in range 20 - 40 years old.
# The age distributions of survived passengers are less than to age distributions of didn't survived passengers. It means many passengers
# in the age range of 20 - 40 years old who didn't survived.

# How about SibSp of survived?
print('\n[16] Siblings/Spouses of survived')
print('Survived by SibSp:', df[df['Survived'] == 1]['SibSp'].count())
print('Not Survived by SibSp:', df[df['Survived'] == 0]['SibSp'].count())
sns.catplot(x='Survived', col='SibSp', data=df, kind='count')
# Based on number of siblings/spouses, many victims didn't aboard with their siblings/spouses or just aboard alone. However, there are 
# some victims that aboard with theirs siblings/spouses. On the other hand, there are still a number of survived passengers either no 
# siblings or no spouses about 342 passengers.

# How about Parch of survived?
sns.catplot(x='Survived', col='Parch', data=df, kind='count')
print('\n[17] Parents/Children of survived')
print('Survived by Parch:', df[df['Survived'] == 1]['Parch'].count())
print('Not Survived by Parch:', df[df['Survived'] == 0]['Parch'].count())
# Based on number of parents/children, many victims didn't aboard with their parents/children or just aboard alone. However, there are
# some victims that aboard with theirs parents/children. On the other hand, there are still a number of survived passengers either no
# parents or no children about 342 passengers.

# How about fare distributions?
plt.figure(7)
plt.hist(df[df['Survived'] == 1]['Fare'], bins=50, edgecolor='black', alpha=0.5, label='Survived')
plt.hist(df[df['Survived'] == 0]['Fare'], bins=25, edgecolor='black', alpha=0.5, label='Not Survived')
plt.xlabel('Fare')
plt.ylabel('count')
plt.title('Fare Distributions of Survived Passengers\nand Not Survived Passengers')
plt.legend()
# Based on Fare, most passengers aboard with fare in range 0 - 50. Many victims number come from that range. However, there are a number
# of survived passengers from that range of fare either. The Fare is medium correlate to Pclass, because first class is high fare (expensive),
# second class is medium fare and third class is low fare (cheap). If We look survived by Pclass, the third class has higher number of
# didn't survived passengers than number of didn't survived passengers in fisrt class where the third class is low fare. Also, the first class
# has higher number of survived passengers than number of survived passengers in third class where the first class is expensive fare.

# Model and Predict
# Add Gender column to represent Sex by binary number, male = 1 and female = 0
df['Gender'] = df['Sex'].replace(to_replace=['male','female'], value=[1,0])
print(df.head())

# Set feature data
X = df[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Gender']]
# Set target data
y = df['Survived']
# Split data to training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
# Logistic Regression
logreg = LogisticRegression()
# Fit the regression
logreg = logreg.fit(X_train, y_train)
# Predict the model by using X_test
y_pred = logreg.predict(X_test)
# Confusiion Matrix
cm = confusion_matrix(y_test, y_pred)
print('\n[18] Confusion Matrix')
print(cm)

plt.figure(8)
sns.heatmap(pd.DataFrame(cm), annot=True, cmap='YlGnBu', fmt='g')
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()

# TN = 119, FP = 20
# FN = 26, TP = 58
# Model predicted 119 passengers who didn't survived that actually didn't.
# Model incorrectly predicted 20 passengers who survived that actually didn't
# Model incorrectly predicted 26 passengers who didn't survived that actually did
# Model predicted 58 passengers who survived that actually did

# Calculate the accuracy, precision and recall
print('\n[19] Evaluation model')
print('Accuracy:', accuracy_score(y_test, y_pred)*100)
print('Mised Classficiation Rate:', 100 - (accuracy_score(y_test, y_pred)*100))
print('Precision:', precision_score(y_test, y_pred)*100)
print('Recall (Positive Rate):', recall_score(y_test, y_pred)*100)
print('Specificity (Negative Rate):', (cm[0,0]/(cm[0,0] + cm[0,1]))*100)
# Accuracy: 79.37219730941703
# Mised Classficiation Rate: 20.627802690582968
# Precision: 74.35897435897436
# Recall: 69.04761904761905
# Specificity: 85.61151079136691

# Percentage score that passengres who actually didn't survived and survived is high enough about 79.37 %.
# About 74.35 % model is correct to predict the passengers who actually survived and about 85.61 % model 
# predict the passengers who didn't survived. The mised classification rate of this model is about 20.62 %. 
# That means, this model is good enough to predict passengers who survived or didn't survived.

# Conclusions 
# The most victims of RMS Titanic are:
# 1. Male
# 2. in third class (Pclass = 3)
# 3. Low fare
#  and the passangers who survived are:
# 1. Female
# 2. in the first class (Pclass = 1)
# 3. High fare

# Based on age of passengers, the higher number of passengers who didn't survived has age in range of 20-40 years old.
# However, in the same range of age, there are a number of passengers who survived although in small numbers.
# Based on model prediction, about 79.37% model predict that the passengers are didn't survived and survived
# where the mised classification rate is about 20.62%. This model is good enough to predict passengers 
# who survived or didn't survived.

plt.show()