
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.cross_validation import train_test_split
from sklearn.svm import LinearSVC #support vector machine classification
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV

in_file = ['18','24','55','60','97','177','185']
dfs= []
for i in in_file:    
    elem = pd.read_csv('FullOct2007'+i+'.csv')    
    dfs.append(elem)
    result = pd.concat(dfs)

labels=['Family & Relationships', 'Entertainment & Music','Society & Culture', 'Computers & Internet']
def data_prepare(file_df,label_list):    
    df = file_df.loc[file_df['maincat'].isin(label_list)][['subject','content','maincat']]
    df = df.replace(np.nan, '', regex=True)    
    df['X'] = df['subject'].map(str) + ' ' +  df["content"].map(str)    
    df['X'].replace(' ', np.nan, inplace=True)    
    df.dropna(subset=['X'], inplace=True)
    initial_y = df['maincat']    
    le = preprocessing.LabelEncoder()    
    le.fit(initial_y)    
    y = le.transform(initial_y)    
    return df.X, y, df
X,y,df = data_prepare(result,labels)

#split data to training dataset and test dataset
X_train, X_test, Y_train, Y_test = train_test_split(df['X'], y, train_size=.75)

#turn data to vector
text_clf_NB = Pipeline([('vect', CountVectorizer(stop_words= 'english')),('tfidf', TfidfTransformer()),('clf', MultinomialNB()),])
text_clf_NB = text_clf_NB.fit(X_train, Y_train)
predicted_NB = text_clf_NB.predict(X_test)
print("original settings for NB model", np.mean(predicted_NB == Y_test))

#grid search for NB
from sklearn.grid_search import GridSearchCV
parameters = {'vect__ngram_range': [(1, 1), (1, 2)],'tfidf__use_idf': (True, False),'clf__alpha': (1,1e-1,1e-2, 1e-3,1e-4),}
gs_clf = GridSearchCV(text_clf_NB, parameters, n_jobs=-1,verbose=10, scoring='log_loss')
gs_clf = gs_clf.fit(X_train, Y_train)
best_parameters, score, _ = max(gs_clf.grid_scores_, key=lambda x: x[1])
for param_name in sorted(parameters.keys()):
    print("%s: %r" % (param_name, best_parameters[param_name]))

#determine best model
new_model = gs_clf.best_estimator_
#make prediction on test dataset
predicted_gs = new_model.predict(X_test)
print("best settings for NB model", np.mean(predicted_gs == Y_test))

