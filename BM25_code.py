
import gensim
from gensim import corpora
import math
import pandas as pd
import numpy as np
import string

class BM25 :
    def __init__(self, fn_docs, delimiter='|') :
        self.dictionary = corpora.Dictionary()
        self.DF = {}
        self.delimiter = delimiter
        self.DocTF = []
        self.DocIDF = {}
        self.N = 0
        self.DocAvgLen = 0
        self.fn_docs = fn_docs
        self.DocLen = []
        self.buildDictionary()
        self.TFIDF_Generator()

    def buildDictionary(self) :
        raw_data = []
        for line in self.fn_docs:
            raw_data.append(line.split(self.delimiter))
        
        self.dictionary.add_documents(raw_data)

    def TFIDF_Generator(self, base=math.e) :
        docTotalLen = 0
        for line in self.fn_docs:
            doc = line.split(self.delimiter)
            docTotalLen += len(doc)
            self.DocLen.append(len(doc))
            #print self.dictionary.doc2bow(doc)
            bow = dict([(term, freq*1.0/len(doc)) for term, freq in self.dictionary.doc2bow(doc)])
            for term, tf in bow.items() :
                if term not in self.DF :
                    self.DF[term] = 0
                self.DF[term] += 1
            self.DocTF.append(bow)
            self.N = self.N + 1
        for term in self.DF:
            self.DocIDF[term] = math.log((self.N - self.DF[term] +0.5) / (self.DF[term] + 0.5), base)
        self.DocAvgLen = docTotalLen / self.N

    def BM25Score(self, Query, k1=1.5, b=0.75) :
        
        query_bow = self.dictionary.doc2bow(Query)
        scores = []
        for idx, doc in enumerate(self.DocTF) :
            commonTerms = set(dict(query_bow).keys()) & set(doc.keys())
            tmp_score = []
            doc_terms_len = self.DocLen[idx]
            for term in commonTerms :
                upper = (doc[term] * (k1+1))
                below = ((doc[term]) + k1*(1 - b + b*doc_terms_len/self.DocAvgLen))
                tmp_score.append(self.DocIDF[term] * upper / below)
            scores.append(sum(tmp_score))
        scores = np.array(scores)/float(len(query_bow))
        return scores

    def TFIDF(self) :
        tfidf = []
        for doc in self.DocTF :
            doc_tfidf  = [(term, tf*self.DocIDF[term]) for term, tf in doc.items()]
            doc_tfidf.sort()
            tfidf.append(doc_tfidf)
        return tfidf

    def Items(self) :
        # Return a list [(term_idx, term_desc),]
        items = self.dictionary.items()
        items.sort()
        return items

#preprocess all files
exclude = set(string.punctuation)
in_file = ['23','25','27','34','36','41','42','55','59','60','77','86','96','161','167','181','183','189','229']
dfs= []
for i in in_file:    
    elem = pd.read_csv('/Users/apple/Desktop/dataset_jirou/FullOct2007'+i+'.csv')    
    dfs.append(elem)
    result = pd.concat(dfs)

labels=['Family & Relationships', 'Entertainment & Music','Society & Culture', 'Computers & Internet']
def test_df(file_df, label_list):
    df = file_df.loc[file_df['maincat'].isin(label_list)][['subject','content','maincat','bestanswer']]
    df = df.replace(np.nan, '', regex=True)    
    df['X'] = df['subject'].map(str) + ' ' +  df["content"].map(str)    
    df['X'].replace(' ', np.nan, inplace=True)    
    df.dropna(subset=['X'], inplace=True)
    for i in range(len(df['X'])):
        
        df['X'].values[i] = ''.join([ch for ch in df['X'].values[i] if ch not in exclude])
    return df.X,df
  

X,df = test_df(result,labels)
df_family = df.loc[df['maincat'].isin(['Family & Relationships'])][['maincat','bestanswer','X']]
df_enter = df.loc[df['maincat'].isin(['Entertainment & Music'])][['maincat','bestanswer','X']]
df_society = df.loc[df['maincat'].isin(['Society & Culture'])][['maincat','bestanswer','X']]
df_computer = df.loc[df['maincat'].isin(['Computers & Internet'])][['maincat','bestanswer','X']]







