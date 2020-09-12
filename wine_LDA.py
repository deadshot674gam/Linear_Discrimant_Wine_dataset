import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

names = ['Class_label','Alcohol','Malic.acid','Ash','Acl','Mg','Phenols','Flavanoids','Nonflavanoid.phenols','Proanth','Color.int','Hue','OD','Proline']

wine = pd.read_csv('wine.csv',names = names,header = 0)

X = wine.iloc[1:,1:13]
y = wine.iloc[1:,0]
corr = wine.corr()
print(corr)
plt.figure(figsize=(13,13))
sns.heatmap(corr);
plt.show()
#splitting dataset into training data and testing data
from sklearn.model_selection import train_test_split as TTS
X_train,X_test,y_train,y_test = TTS(X,y,test_size = 0.2, random_state = 0)

#feature scaling using standardscaler
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train,y_train)
X_test = sc.transform(X_test)

#performing Linear Discriminant Analysis 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 1)
X_train = lda.fit_transform(X_train,y_train)
X_test = lda.transform(X_test)

#using RandomForestClassifier to train the model and predict 
from sklearn.ensemble import RandomForestClassifier as RDF
classifier = RDF(max_depth = 2 , random_state = 0)
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)


print("misclassified labels : ",(y_pred!=y_test).sum())
#visualisation
#plotting tested data
plt.scatter(X_test, y_test)
plt.scatter(X_test, y_pred)
plt.show()

error = y_pred - y_test
l = np.array(range(36))
plt.scatter(error,l)
plt.show()


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

#accuracy
accuracy = accuracy_score(y_test,y_pred)
print(accuracy)
#confusion matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)
plt.figure(figsize=(3,3))
sns.heatmap(cm);
