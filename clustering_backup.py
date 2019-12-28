import nltk
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.cluster import DBSCAN
import postgresql

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

#Считываем конфиг
file_conf = open("config.txt","r")
conf_text = file_conf.read()
path_to_news = re.findall('PATH_TO_NEWS = \[(.*?)\]', conf_text)[0]
postgres_connection =  re.findall('POSTGRES_CONNECT = \[(.*?)\]', conf_text)[0]
file_conf.close()
print(postgres_connection)

fl = os.listdir(path_to_news)
files_lis = list(fl)


#Проверка на валидные названия файлов
for nm in files_lis:
    if (re.match(r'.+\..{3}', nm) == None):
        files_lis.remove(nm)
files_list = sorted(files_lis)


totalvocab_stem = []
totalvocab_token = []
all = []
all_texts = []


for i, f_name in enumerate(files_list):
    totalvocab_stem = []
    totalvocab_token = []
    f = open(path_to_news + f_name, "r")
    titles = f.read()
    if(i == len(files_list) - 1 or i == len(files_list) - 2):
        print(f_name)
    all_texts.append(titles)
    allwords_stemmed = token_and_stem(titles)
    totalvocab_stem.extend(allwords_stemmed)
    allwords_tokenized = token_only(titles)
    totalvocab_token.extend(allwords_tokenized)
    tutu = ''
    tutu += ','.join(totalvocab_stem)
    all.append(tutu)
    f.close()
all_texts = sorted(list(set(all_texts)))


tfidf_vectorizer = TfidfVectorizer( max_features=2500,stop_words=stopwords)
X = tfidf_vectorizer.fit_transform(all_texts)
clustering = DBSCAN(eps=0.965, min_samples=1,metric = 'euclidean').fit(X)

print(clustering.labels_)
k = list(clustering.labels_)
у_db = clustering.fit_predict(X)
print(у_db)
news_new = []
mapp_news = dict()
clasters = set()


for i, o in enumerate(k):
    for j in news_new:
        if(j == o):
            clasters.add((o))
    news_new.append(o)
print(clasters)

_texts = {}

for j in clasters:
    _texts[j] = []
    for i, o in enumerate(k):
        if j == o:
            _texts[j].append(all_texts[i])


db = postgresql.open(postgres_connection)
for key in _texts:
    text_in_db = ''
    text_in_db = '\n'.join(_texts[key])
    try:
        ins = db.prepare("insert into news_list.world (\"NEWSTEXT\") VALUES ($1)")
        do_insert = ins(re.sub(r'\[||\]', '', text_in_db))
    except:
        print("An exception occurred")
