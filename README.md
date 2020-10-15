# Titanic Analysis
## Objective
Predict the survival of the passengers aboard RMS Titanic.
## Features
1. PassengersId
2. Survived (Survived = 1, didn't Survived = 0)
3. Pclas (Passanger Class - Pclass = 1, Pclass = 2, Pclass = 3)
4. Name
5. Age
6. Sex (Female and Male)
7. SibSp (Number of Siblings/Spouses)
8. Parch (Number of Parents/Children)
9. Ticket
10. Fare
11. Cabin
12. Embarked
## Import Library
```
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
```
## Load Data
```
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
```
##### Results
```
[4] Total missing value:
Each item: PassengerId      0
Survived         0
Pclass           0
Name             0
Sex              0
Age            177
SibSp            0
Parch            0
Ticket           0
Fare             0
Cabin          687
Embarked         2
dtype: int64
Total : 866
```
We get total 866 missing value, there are in Age (177 data), Cabin (687 data) and Embarked (2 data). The Cabin has high missing value, so the Cabin is removed from dataframe. PassengersId and Ticket are removed too.
## Remove Uncessery Features
```
# Remove Cabin, PassengerId, and Ticket from dataframe
df = df.drop(['Cabin','PassengerId', 'Ticket'], axis=1)
print('\n[5] After remove Cabin, PassengerId and Ticket:')
print(df.info())
```
## Fill The Missing Value
- The missing value in Age column is replaced by mean of Age.
- Two values are missing in Embarked data. To fill these missing values, We assume and replace by high fequency of Embarked.
```
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
# From these results, the S has high frequency of embarkation. Then, We replace the missing values by S emberkation
df['Embarked'].fillna('S', inplace = True)
print(df.info())
```
##### Results
```
# Embarked
# C    168
# Q     77
# S    644
```
From these results, the S is high frequency of embarkation. Then, We replace the missing values by S emberkation.
## Short Describe and Correlation
Based on the main objective, We can generate a question that are there correlation between survived and other features?
```
# Short describe
print('\n[8] Describe data:')
print(df.describe())

correaltion = df.corr()
print('\n[9] Correlation for each pair feature:')
print(correaltion)
```
##### Results
```
[8] Describe data:
         Survived      Pclass         Age       SibSp       Parch        Fare
count  891.000000  891.000000  891.000000  891.000000  891.000000  891.000000
mean     0.383838    2.308642   29.699118    0.523008    0.381594   32.204208
std      0.486592    0.836071   13.002015    1.102743    0.806057   49.693429
min      0.000000    1.000000    0.420000    0.000000    0.000000    0.000000
25%      0.000000    2.000000   22.000000    0.000000    0.000000    7.910400
50%      0.000000    3.000000   29.699118    0.000000    0.000000   14.454200
75%      1.000000    3.000000   35.000000    1.000000    0.000000   31.000000
max      1.000000    3.000000   80.000000    8.000000    6.000000  512.329200

[9] Correlation for each pair feature:
          Survived    Pclass       Age     SibSp     Parch      Fare
Survived  1.000000 -0.338481 -0.069809 -0.035322  0.081629  0.257307
Pclass   -0.338481  1.000000 -0.331339  0.083081  0.018443 -0.549500
Age      -0.069809 -0.331339  1.000000 -0.232625 -0.179191  0.091566
SibSp    -0.035322  0.083081 -0.232625  1.000000  0.414838  0.159651
Parch     0.081629  0.018443 -0.179191  0.414838  1.000000  0.216225
Fare      0.257307 -0.549500  0.091566  0.159651  0.216225  1.000000
```
In generally, the survived is inversely correlated with Pclass (Passanger Class) as a higher correlation from other features. It means the passangers who survived are in the fisrt class (Pclass = 1), where the first class indicates high socio-economic status. The Pclass also inversely correlated with Fare where the first class has high fare (expensive) and third class has low fare (cheap). It means, the passengers who survived are domainted in first class. On the other hand, the survived is weak correlated to Age, SibSp (Siblings/Spouses) and Parch (Parrents/Children).
## How much passenger who aboard in RMS Titanic by Gender?
```
print('\n[10] Passanger who aboard in RMS titanic by Gender')
print('Female:', df[df['Sex'] == 'female']['Sex'].count())
print('Male:', df[df['Sex'] == 'male']['Sex'].count())
```
##### Result
```
[10] Passanger who aboard in RMS titanic by Gender
Female: 314
Male: 577
```
## How much passengers are survived by overall?
```
print('\n[11] Passanger who survived and did not survived')
print('Survived:', df[df['Survived'] == 1]['Survived'].count())
print('No Survived:', df[df['Survived'] == 0]['Survived'].count())

ax = sns.countplot(x = 'Survived', data = df)
ax.set_title('Number of Survived and didn\'t Survived')
```
##### Result
```
[11] Passanger who survived and did not survived
Survived: 342
No Survived: 549
```
![Figure1. Survived](https://github.com/auliakhalqillah/Titanic_Analysis/blob/main/titanic_survived_1.png)
## How much passengers are survived and aren't survived by Gender?
```
print('\n[12] Passanger who survived by Gender')
print('Female :',df[(df['Survived'] == 1) & (df['Sex'] == 'female')]['Survived'].count())
print('Male :',df[(df['Survived'] == 1) & (df['Sex'] == 'male')]['Survived'].count())

print('\n[13] Passanger who did not survived by Gender')
print('Female :',df[(df['Survived'] == 0) & (df['Sex'] == 'female')]['Survived'].count())
print('Male :',df[(df['Survived'] == 0) & (df['Sex'] == 'male')]['Survived'].count())

sns.catplot(x='Survived', col='Sex', kind='count', data = df)
```
##### Result
```
[12] Passanger who survived by Gender
Female : 233
Male : 109

[13] Passanger who did not survived by Gender
Female : 81
Male : 468
```
![Figure2. Gender](https://github.com/auliakhalqillah/Titanic_Analysis/blob/main/titanic_sex_2.png)
## How much passangers are survived by Pclass?
```
print('\n[14] Passanger who survived and did not survived by Pclass')
print(df.groupby(['Survived','Pclass']).size())

sns.catplot(x='Survived',col='Pclass',data=df,kind='count')
```
##### Result
```
[14] Passanger who survived and did not survived by Pclass
Survived  Pclass
0         1          80
          2          97
          3         372
1         1         136
          2          87
          3         119
dtype: int64
```
![Figure3. Pclass](https://github.com/auliakhalqillah/Titanic_Analysis/blob/main/titanic_pclass_3.png)

RMS Titanic was boarded 891 passengers consisting of 314 female passengers and 577 male passengers. Based on these data, about 342 passengers are survived and 549 passengers are not survived. By 342 passengers are survived, as many as 233 passngers are female and 109 passengers are male. By 549 passengers are not survived, as many as 81 passengers are female and 468 passengers are male. Female passengers are high number of survived than male passengers. Furthermore, the high number of passangers who survived are in the first class about 136 passengers and the high number of passengers who didn't survived about 372 passengers are in third class.

## How Age distribution in RMS Titanic?
```
print('\n[15] Age distribution in RMS Titanic')
plt.figure(4, figsize=(10,6))
plt.hist(df['Age'], bins=10, edgecolor='black',label='Age distribution in RMS Titanic',alpha=0.5)
plt.hist(df[df['Survived'] == 1]['Age'], bins=10, edgecolor='black',label='Age distribution of survived',alpha=0.5)
plt.hist(df[df['Survived'] == 0]['Age'], bins=10, edgecolor='black',label='Age distribution of not survived',alpha=0.5)
plt.xlabel('Age')
plt.ylabel('count')
plt.title('Age Distributions')
plt.legend()
```
![Figure4. Age](https://github.com/auliakhalqillah/Titanic_Analysis/blob/main/titanic_age_4.png)

Based on Age distributions, the passengers who aboard in RMS Titanic are dominated by passengers who has age in range 20 - 40 years old. The age distributions of survived passengers are less than to age distributions of didn't survived passengers. It means many passengers in the age range of 20 - 40 years old who didn't survived.

## How much SibSp of survived?
```
print('\n[16] Siblings/Spouses of survived')
print('Survived by SibSp:', df[df['Survived'] == 1]['SibSp'].count())
print('Not Survived by SibSp:', df[df['Survived'] == 0]['SibSp'].count())

sns.catplot(x='Survived', col='SibSp', data=df, kind='count')
```
##### Result
```
[16] Siblings/Spouses of survived
Survived by SibSp: 342
Not Survived by SibSp: 549
```
![Figure5. SibSp](https://github.com/auliakhalqillah/Titanic_Analysis/blob/main/titanic_sibsp_5.png)

Based on number of siblings/spouses, many victims didn't aboard with their siblings/spouses or just aboard alone. However, there are some victims that aboard with theirs siblings/spouses. On the other hand, there are still a number of survived passengers either no siblings or no spouses about 342 passengers.

## How much Parch of survived?
```
print('\n[17] Parents/Children of survived')
print('Survived by Parch:', df[df['Survived'] == 1]['Parch'].count())
print('Not Survived by Parch:', df[df['Survived'] == 0]['Parch'].count())

sns.catplot(x='Survived', col='Parch', data=df, kind='count')
```
##### Result
```
[17] Parents/Children of survived
Survived by Parch: 342
Not Survived by Parch: 549
```
![Figure6. Parch](https://github.com/auliakhalqillah/Titanic_Analysis/blob/main/titanic_parch_6.png)

Based on number of parents/children, many victims didn't aboard with their parents/children or just aboard alone. However, there are some victims that aboard with theirs parents/children. On the other hand, there are still a number of survived passengers either no parents or no children about 342 passengers.

## How about fare distributions?
```
plt.figure(7)
plt.hist(df[df['Survived'] == 1]['Fare'], bins=50, edgecolor='black', alpha=0.5, label='Survived')
plt.hist(df[df['Survived'] == 0]['Fare'], bins=25, edgecolor='black', alpha=0.5, label='Not Survived')
plt.xlabel('Fare')
plt.ylabel('count')
plt.title('Fare Distributions of Survived Passengers\nand Not Survived Passengers')
plt.legend()
```
![Figure7. Fare](https://github.com/auliakhalqillah/Titanic_Analysis/blob/main/titanic_fare_7.png)

Based on Fare, most passengers aboard with fare in range 0 - 50. Many victims number come from that range. However, there are a number of survived passengers from that range of fare either. The Fare is medium correlate to Pclass, because first class is high fare (expensive), second class is medium fare and third class is low fare (cheap). If We look survived by Pclass, the third class has higher number of didn't survived passengers than number of didn't survived passengers in fisrt class where the third class is low fare. Also, the first class has higher number of survived passengers than number of survived passengers in third class where the first class is expensive fare.

# Model and Predict
Add Gender column to represent Sex by binary number, male = 1 and female = 0
```
df['Gender'] = df['Sex'].replace(to_replace=['male','female'], value=[1,0])
print(df.head())
```
##### Result
```
Survived  Pclass                                               Name     Sex   Age  SibSp  Parch     Fare Embarked  Gender
0         0       3                            Braund, Mr. Owen Harris    male  22.0      1      0   7.2500        S       1
1         1       1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1      0  71.2833        C       0
2         1       3                             Heikkinen, Miss. Laina  female  26.0      0      0   7.9250        S       0
3         1       1       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1      0  53.1000        S       0
4         0       3                           Allen, Mr. William Henry    male  35.0      0      0   8.0500        S       1
```

```
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
```
##### Result
```
[18] Confusion Matrix
[[119  20]
 [ 26  58]]
```
![Figure 8. Confusion Matirx](https://github.com/auliakhalqillah/Titanic_Analysis/blob/main/titanic_cm_8.png)

```
TN = 119, FP = 20
FN = 26, TP = 58
```
- Model predicted 119 passengers who didn't survived that actually didn't.
- Model incorrectly predicted 20 passengers who survived that actually didn't
- Model incorrectly predicted 26 passengers who didn't survived that actually did
- Model predicted 58 passengers who survived that actually did

## Calculate the accuracy, precision and recall
```
print('\n[19] Evaluation model')
print('Accuracy:', accuracy_score(y_test, y_pred)*100)
print('Mised Classficiation Rate:', 100 - (accuracy_score(y_test, y_pred)*100))
print('Precision:', precision_score(y_test, y_pred)*100)
print('Recall (Positive Rate):', recall_score(y_test, y_pred)*100)
print('Specificity (Negative Rate):', (cm[0,0]/(cm[0,0] + cm[0,1]))*100)
```
###### Result
```
[19] Evaluation model
Accuracy: 79.37219730941703
Mised Classficiation Rate: 20.627802690582968
Precision: 74.35897435897436
Recall (Positive Rate): 69.04761904761905
Specificity (Negative Rate): 85.61151079136691
```

Percentage score that passengres who actually didn't survived and survived is high enough about 79.37 %. About 74.35 % model is correct to predict the passengers who actually survived and about 85.61 % model predict the passengers who didn't survived. The mised classification rate of this model is about 20.62 %. That means, this model is good enough to predict passengers who survived or didn't survived.

# Conclusions 
The most victims of RMS Titanic are:
1. Male
2. in third class (Pclass = 3)
3. Low fare
and the passangers who survived are:
1. Female
2. in the first class (Pclass = 1)
3. High fare

Based on age of passengers, the higher number of passengers who didn't survived has age in range of 20-40 years old. However, in the same range of age, there are a number of passengers who survived although in small numbers. Based on model prediction, about 79.37% model predict that the passengers are didn't survived and survived where the mised classification rate is about 20.62%. This model is good enough to predict passengers who survived or didn't survived.

# Source
https://www.kaggle.com/c/titanic/data
