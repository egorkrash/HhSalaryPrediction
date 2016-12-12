import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from joblib import Parallel, delayed
import re
import pymorphy2
import multiprocessing
from tqdm import tqdm
import pickle
from keras.preprocessing.sequence import pad_sequences

tokenizer = pickle.load(open('TOKENIZER.pkl','rb'))
morph = pymorphy2.MorphAnalyzer()

def _word2canonical4w2v(word):
    elems =  morph.parse(word)
    my_tag = ''
    res = []
    for elem in elems:
            if 'VERB' in elem.tag or 'GRND' in elem.tag or 'INFN' in elem.tag:
                    my_tag = 'V'
            if 'NOUN' in elem.tag:
                    my_tag = 'S'
            normalised = elem.normalized.word
            if my_tag == '':
                    res.append((normalised, ''))
            res.append((normalised, my_tag))
    tmp = list(filter(lambda x: x[1] != '', res))
    if len(tmp) > 0:
        return tmp[0]
    else:
        return res[0]


def word2canonical(word):
    return  _word2canonical4w2v(word)[0]


def getWords(text, filter_short_words=False):
    if filter_short_words:
        return filter(lambda x: len(x) > 3, re.findall(r'(?u)\w+', text))
    else:
        return re.findall(r'(?u)\w+', text)


def text2canonicals(text, add_word=False, filter_short_words=True):
    words = []
    for word in getWords(text, filter_short_words = filter_short_words):
        words.append(word2canonical(word.lower()))
        if add_word:
            words.append(word.lower())
    return words


def remove_tags(text):
    soup = BeautifulSoup(text,'html.parser')
    return soup.text


def convert_text(text):
    text = text2canonicals(remove_tags(text).lower())
    return text


def parallelization(massive,tq = True):    
    num_cores = 20#multiprocessing.cpu_count()
    if tq:
        results = np.array(Parallel(n_jobs=num_cores)(delayed(convert_text)(i) for i in tqdm(massive)))
        return results
    else:
        results = Parallel(n_jobs=num_cores)(delayed(convert_text)(i) for i in massive)
        return results

    
def make_classes(y, nb_classes):
    chunk_size = 440000/nb_classes
    y = np.array(map(int,y))/chunk_size
    return y    


def make_dataset(path,df=True,x=False,y=False, chunks = False):
    data = pd.read_csv(path,index_col=False)
    data = data[data['salary:currency'] == 'RUR'] 
    data = data[((data['salary:to'] > 100) & (data['salary:to'] <  440000)) 
                & ((data['salary:from'] > 100) & (data['salary:to'] <  440000))]
    
    to_return = []
    
    if df == True:
        to_return.append(data)
    
    if x == True:
        texts = parallelization(data.description.values,True)
        texts = map(lambda x: ' '.join(x),texts) 
        texts = map(lambda x: x.encode('utf8'), texts)

        sequences = tokenizer.texts_to_sequences(texts) 
        X = pad_sequences(sequences, maxlen=200) 
        to_return.append(X)


    if y == True:
        sal_to = data['salary:to']; sal_from = data['salary:from']
        Y = np.array([max(sal_to.iloc[i],sal_from.iloc[i]) for i in range(len(sal_to))])
        
        if chunks != False:
            Y = make_classes(Y,chunks)
            
        to_return.append(Y)
        

    return tuple(to_return)

