import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import string
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

ls=WordNetLemmatizer()

df=pd.read_csv(r"D:\datasets\SPAM_HAM\spam.csv",encoding="latin-1")
df=df.rename(columns={'v1':'Label','v2':'Message'})

df=df.drop({'Unnamed: 2','Unnamed: 3','Unnamed: 4'},axis=1)


cor=[]
for i in range(0,len(df['Message'])):
    message=re.sub('[^a-zA-Z]',' ',df['Message'][i])
    message=df["Message"][i].lower()
    message=message.split(' ')
    message=[ls.lemmatize(word) for word in message if not word in stopwords.words('english')]
    message=' '.join(message)
    cor.append(message)


cv=CountVectorizer()
X=cv.fit_transform(cor).toarray()

l=[]
d=cv.vocabulary_
for i in d.keys():
    if not i.isalnum():
        l.append(i)
l

y=pd.get_dummies(df['Label'])
y=y.iloc[:,-1].values

df['mdg']=df['Message'].apply(clean)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)

spam_detector=MultinomialNB().fit(X_train,y_train)

spam_detector1=LogisticRegression().fit(X_train,y_train)

spam_detector2=RandomForestClassifier().fit(X_train,y_train)

spam_detector3=MLPClassifier().fit(X_train,y_train)


ypred=spam_detector.predict(X_test)

ypred1=spam_detector1.predict(X_test)

ypred2=spam_detector2.predict(X_test)

ypred3=spam_detector3.predict(X_test)





import joblib      
import pickle
pickle.dump(spam_detector,open('D:\datasets\SPAM_HAM\spam_detector.pickle','wb'))
pickle.dump(cv,open('D:\datasets\SPAM_HAM\CV.pickle','wb'))
model = pickle.load(open('D:\datasets\SPAM_HAM\spam_detector.pickle', 'rb'))
CVM=pickle.load(open('D:\datasets\SPAM_HAM\CV.pickle', 'rb'))

ex=CVM.transform(["Sorry my roommates took forever, it ok if I come by now?"])
model.predict(ex)[0]

