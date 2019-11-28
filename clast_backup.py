import numpy as np
import pandas as pd
import nltk
import re
import os
import codecs
#from sklearn import feature_extraction
'''import mpld3
import matplotlib.pyplot as plt
import matplotlib as mpl'''
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.stem.snowball import SnowballStemmer
nltk.download('punkt')
nltk.download('stopwords')
stemmer = SnowballStemmer("russian")
stopwords = nltk.corpus.stopwords.words('russian')
stopwords.extend(['что', 'это', 'так', 'вот', 'быть', 'как', 'в', 'к', 'на', 'о', 'этот', 'эти', 'которые', 'меня'\
                  'тот', 'те', 'такие', 'тех', 'их', 'там', 'тут', 'который', 'до', 'эта', 'этот', 'эт'\
                  ,'сказ', 'котор', 'кто', 'кого', 'когда', 'тогда', 'она', 'оно', 'его', 'того', 'ты' , 'официальн', 'для', 'можно', 'сказать'\
                  , 'будут', 'будет', 'сказа', 'какой', 'какие', 'какая', 'минут', 'смотреть', 'Для', 'сказал', 'сказа','сказали','сказала','через' \
                     , 'все', 'вся', 'всех', 'он', 'я', 'заяв', 'наш', 'ваш', 'для', 'ее','ей', 'им', 'могут', 'может',
                  'свои', 'свой', 'сво', 'оста', 'мог', 'до', 'также', 'тоже', 'то', 'же', 'возле', 'около', 'как', 'сообщ', 'котор', 'после', 'потом', \
                  'минут', 'наход', 'которая', 'которую', 'которых', 'которым', 'котор', 'они' , 'новост', 'вид', 'виды', 'заяв', 'заявил' \
                                                                                                                                   'своя', 'своим', 'своих', 'своего', 'позвол', 'счита', 'где', 'пишет', ' позвол', 'оста', 'врем', \
                  'заявил'])

def token_and_stem(text):
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    for token in tokens:
        a = 0
        for w in stopwords:
            if (token == w):
                a = 1
        if re.search('[а-яА-Я]', token) and a != 1:
            filtered_tokens.append((re.sub('\W*||\-','', token)))
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


def token_only(text):
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    for token in tokens:
        a = 0
        for w in stopwords:
            if (token == w):
                a = 1
        if re.search('[а-яА-Я]', token) and a != 1:
            filtered_tokens.append(re.sub('\W*||\-','', token))
    return filtered_tokens


# Создаем словари (массивы) из полученных основ
#from gensim.models import doc2vec
from collections import namedtuple
fl = os.listdir("/Users/amalakhova/Documents/news_site/parser/events/event/")
files_lis = list(fl)
files_lis.remove('.DS_Store')
files_list = sorted(files_lis)
#files_list = ['Worldscienceozone-hole-over-antarctica-shrinks-to-record-small-size_ru.txt']
analyzedDocument = namedtuple('analyzeddocument', 'words tags')
totalvocab_stem = []
totalvocab_token = []
all = []
all_texts = []
for i, f_name in enumerate(files_list):
    totalvocab_stem = []
    totalvocab_token = []
    f = open('/Users/amalakhova/Documents/news_site/parser/events/event/'+ f_name, "r")
    titles = f.read()#.encode('windows-1252')
    if(i == len(files_list) - 1 or i == len(files_list) - 2):
        print(f_name)
    '''if(f_name == 'Worldhttps/www.foxnews.comworldisis-al-baghdadi-underwear-spy-dna-test-kurds_ru.txt' or \
            f_name == 'Worldhttps/ria.ru201910271560275708.html_ru.txt'):
        print(i)'''
    all_texts.append(titles)
    allwords_stemmed = token_and_stem(titles)
    #print(allwords_stemmed)
    totalvocab_stem.extend(allwords_stemmed)
    allwords_tokenized = token_only(titles)
    totalvocab_token.extend(allwords_tokenized)
#print(totalvocab_token)
    #all.append((analyzedDocument(totalvocab_stem,i)))
    tutu = ''
    tutu += ','.join(totalvocab_stem)
    all.append(tutu)
    f.close()
all_texts = sorted(list(set(all_texts)))

#print(totalvocab_stem)
#print(all_texts[106])
print('_______________________')
#print(all_texts[148])
#n_featur=200000
tfidf_vectorizer = TfidfVectorizer( max_features=2500,stop_words=stopwords)
X = tfidf_vectorizer.fit_transform(all_texts)
#print(X[159])
from sklearn.metrics.pairwise import linear_kernel
cosine_similarities = linear_kernel(X[0], X).flatten()

#160 145 143 133 128 129 154 130 144 107 106 113 147 150 148 135 155 159
#print(cosine_similarities) расстояние ближайшие новости
related_docs_indices = cosine_similarities.argsort()[:-20:-1]
#print(related_docs_indices) расстояние

from sklearn.datasets import fetch_20newsgroups
twenty = fetch_20newsgroups()
#print(twenty.data)
tfidf = TfidfVectorizer().fit_transform(twenty.data)
#print(tfidf[1:2])
from sklearn.metrics.pairwise import linear_kernel
cosine_similarities = linear_kernel(tfidf[0:1], tfidf).flatten()
related_docs_indices = cosine_similarities.argsort()[:-5:-1]
#print(cosine_similarities[related_docs_indices])


#print(X)

from sklearn.cluster import DBSCAN
#X = np.array(X)#.todense()


from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
'''
pca = PCA(n_components=2).fit(X)
data2D = pca.transform(X)
plt.scatter(data2D[:,0], data2D[:,1], c=data.target)
plt.show()
'''
'''colors = np.random.rand(20)

plt.scatter(X, a, c=colors, s=100, alpha=0.65, marker=(5, 0))
plt.show()'''
clustering = DBSCAN(eps=0.965, min_samples=1,metric = 'euclidean').fit(X)
print(clustering.labels_)
k = list(clustering.labels_)
у_db = clustering.fit_predict(X)
print(у_db)
'''plt.scatter(X[у_db==0,0],
            X[у_db==0,-1],
            c='lightblue',
            marker= 'o',
            s=40,
            lаbеl= 'кластер 1')
plt.scatter(X[у_db==-1,0],
            X[у_db== -1,-1],
            c='red',
            marker='s',
            s=40,
            lаЬеl='кластер 2')

plt.legend()
plt.show()'''
news_new = []
mapp_news = dict()
clasters = set()
for i, o in enumerate(k):
    #if(o == 1):
        #print(all_texts[i])
        #print(k.index(o))
    for j in news_new:
        if(j == o):
            clasters.add((o))
          #  mapp_news[o] = all_texts[i]
           # print(k.index(o))
            #print(o)
           # f = open('/Users/amalakhova/Documents/news_site/parser/events/'+str(o) + '.txt', "a")
            #for x in all_texts[i]:
               # f.write(str(x))
            #f.close()
    news_new.append(o)
print(clasters)
for j in clasters:
    for i, o in enumerate(k):
        if j == o:
            f = open('/Users/amalakhova/Documents/news_site/parser/events/' + str(o) + '.txt', "a")
            # for x in all_texts[i]:
            # f.write(str(x))
            print(o)
            #print(all_texts[i])
            f = open('/Users/amalakhova/Documents/news_site/parser/events/' + 'News2' + '.txt', "a")
            f.write('ioioioioiiioioi' + str(o) + 'okokokokokkokokokokok')
            for x in all_texts[i]:
                f.write(str(x))
            f.write('plplplplplpplplplp')
            #f.write('\n')
            f.close()

#first_vector_tfidfvectorizer=X[0]

#df = pd.DataFrame(first_vector_tfidfvectorizer.T.todense(), index=tfidf_vectorizer.get_feature_names(), columns=["tfidf"])
#df.sort_values(by=["tfidf"],ascending=False)
#print(df)

#print(tfidf_matrix.shape)
#print(tfidf_vectorizer.get_feature_names())
#from scipy.cluster.hierarchy import linkage, dendrogram
'''f = open('v_.txt', "w+")
for x in X:
    print(x)
    f.write(str(x))
f.close()
'''
'''


from sklearn.cluster import KMeans

km = KMeans(n_clusters=8)
km.fit(tfidf_matrix)
idx = km.fit(tfidf_matrix)
clusters = km.labels_.tolist()

print(clusters)
print (km.labels_)'''