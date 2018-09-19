import numpy as np
import pandas as pd
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC #support vector machine classification
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

in_file = ['2','8','105','115','131','229']
dfs= []
for i in in_file:
    elem = pd.read_csv('data/FullOct2007'+i+'.csv')
    dfs.append(elem)
result = pd.concat(dfs)
    
df = result.loc[result['maincat'].isin(['Family & Relationships', 'Entertainment & Music', 
                                        'Society & Culture', 'Computers & Internet'])][['subject','content','maincat']]

labels= ['Family & Relationships', 'Entertainment & Music', 
            'Society & Culture', 'Computers & Internet']
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
    df['y'] = pd.Series(y, index = df.index)
    return df.X, y, df

from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(df['X'], df['y'], train_size=.75)


