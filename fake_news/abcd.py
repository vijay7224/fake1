import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

#-----data load-------
data=pd.read_csv("C:\\Users\\payal\\Desktop\\training datasets\\True.csv")
data1=pd.read_csv("C:\\Users\\payal\\Desktop\\training datasets\\Fake.csv")


dataset=data[["title"]]
dataset["label"]=1

dataset1=data1[["title"]]
dataset1["label"]=0

#__----- data merge----\
datasets=pd.concat([dataset,dataset1])

# data divide into x and y 
X = datasets['title']    
y = datasets['label'] 


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

ve = CountVectorizer(stop_words='english')
X_trains = ve.fit_transform(X_train)
X_tests = ve.transform(X_test)

model = LogisticRegression()
model.fit(X_trains, y_train)
model.score(X_tests,y_test)*100

# model dump
joblib.dump(model, 'model')
joblib.dump(ve, 'vectorizer')