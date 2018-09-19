from PorterStemmer import PorterStemmer
import numpy as np
from numpy import linalg as la
import math
import re
import os
from collections import Counter
def dotProduct(d1, d2):
    if len(d1) < len(d2):
        return dotProduct(d2, d1)
    else:
        return sum(d1.get(f, 0) * v for f, v in d2.items())

def dotProduct_s(d1, d2):
    if len(d1) < len(d2):
        return dotProduct(d2, d1)
    else:
        maxi = 0
        result = 0
        for f,v in d2.items():
            t = d1.get(f,0) * v
            result += t
            maxi = max(0, abs(t))
        return result, maxi

def multiply(scale, d):
    d = dict(map(lambda (k, v): (k, v*scale), d.iteritems()))
    return d  

def increment(d1, scale, d2):
    for f, v in d2.items():
        d1[f] = d1.get(f, 0) + v * scale

def initial_center(k, TFIDF):
	center = [[0]*len(TFIDF[-1]) for i in range(k)]
	for i in range(len(TFIDF[-1])):
		group = [TFIDF[j][i] for j in range(len(TFIDF))]
		mini, maxi = min(group), max(group)
		s = np.random.uniform(mini,maxi,k)
		for index in range(k):
			center[index][i] = s[index]
	return center	

def cal_dis(a,b):
	upper = np.dot(a,b)
	norm_a = sum([i**2 for i in a])
	norm_a = np.sqrt(norm_a)
	norm_b = sum([i**2 for i in b])
	norm_b = np.sqrt(norm_b)
	down = norm_a*norm_b
	return 1- upper/float(down)

def kmeans_clustering(TFIDF,center):
	partition = [0]*len(TFIDF)
	distance = [0]*len(center)
	flag = True
	while flag:
		flag = False
		for j in range(len(TFIDF)):
			for i in range(len(center)):
				distance[i] = cal_dis(center[i],TFIDF[j])
			partition[j] = distance.index(min(distance))
		center_new = list()
		for j in range(len(center)):
			partition_j = [TFIDF[i] for i in range(len(partition)) if partition[i]==j]
			new_center_j = [sum(i)/len(partition_j) for i in zip(*partition_j)]
			center_new.append(new_center_j)		
		if center_new != center:
			center = center_new
			flag = True
	return partition

def get_stopwords(file = 'stopword.txt'):
	noise_file = open(file,'r').readlines()
	noise_words_set = list()
	for line in noise_file:
		noise_words_set += filter(None, re.split("\W*\d*",line))
	return noise_words_set

def read_txt(path="dataset"):
	for file in os.listdir(path):
	    if file.endswith("ge.txt"):	    	
	    	file = os.path.join(path,file)
	        input_file = open(file,'r')
	return input_file

def remove_porterstemmer(input_file,noise_words_set):
	questions = list()
	word_weight = []
	p = PorterStemmer()
	for line in input_file:
		line = line.lower()
		words = filter(None, re.split("\W*\d*", line))
		question = []
		for word in words:
			new_word = p.stem(word,0,len(word)-1)
			if new_word not in noise_words_set and len(new_word)>2:
				question.append(new_word)
		questions.append(question)
		word_weight.append(Counter(question))
	return word_weight, questions

def tfidf(word_weight):
    all_words = {}
    tfidf = []
    N = len(word_weight)
    for tf in word_weight:
        tfidf.append(tf)
        for f, v in all_words.items():
            v[1] = False
        for word in tf:
            if word not in all_words:
                all_words[word]  = [1, True]
            elif not all_words[word][1]:
                all_words[word][0] += 1
                all_words[word][1] = True
    for tf in tfidf:
        for f in tf:
            tf[f] *= (1 + math.log(float(N) / all_words.get(f,0)[0]))
    return tfidf

def main():
	input_file = read_txt()
	lines = input_file.readlines()
	noise_words_set = get_stopwords()
	word_weight, questions = remove_porterstemmer(lines,noise_words_set)
	TFIDF = tfidf(word_weight)
	print TFIDF[0]
	input_file.close()

if __name__ == "__main__":
    main()
# N = 4
# initial_center = initial_center(N, TFIDF)
# cluster_label = kmeans_clustering(TFIDF, initial_center)
# cluster_name = [[] for i in range(N)]
# name_of_cluster = [[list(),0] for i in range(N)]
# for i in xrange(len(cluster_label)):
# 	label = cluster_label[i]
# 	cluster_name[label].append(names[i][0])
# 	if name_of_cluster[label][1] <= max(TFIDF[i]):
# 		name_index = TFIDF[i].index(max(TFIDF[i]))
# 		word = words[name_index]
# 		word = map_words[word][0]
# 		if word not in name_of_cluster[label][0]:
# 			if name_of_cluster[label][1] < max(TFIDF[i]):
# 				name_of_cluster[label][0] = [word]
# 				name_of_cluster[label][1] = max(TFIDF[i])
# 			else:
# 				name_of_cluster[label][0].append(word)

# output_file = open("output.txt",'w')
# for i in range(N):
# 	title = " ".join(name_of_cluster[i][0])
# 	clause = ", ".join(cluster_name[i])
# 	output_file.write(title+": "+clause+'\n')
# output_file.close()









	
