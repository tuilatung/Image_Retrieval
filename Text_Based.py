import os
import shutil
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.image as mpimg
import time
import string
import nltk
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')


IMAGE_DIR = './dataset/img'
LABEL_DIR = './dataset/txt'

"""Move all files from source folder to destionation folder"""
def move_all(source_dir, destination_dir, ftype='all'):
    file_names = os.listdir(source_dir)
    types = ['txt', 'png']
    if ftype == 'all':
        for file_name in file_names:
            shutil.move(os.path.join(source_dir, file_name), destination_dir)
        return
    if ftype not in types:
        print(f"File endwith {ftype} doesn't support!")
        return
    for file_name in file_names:
        if file_name.endswith(ftype):
            shutil.move(os.path.join(source_dir, file_name), destination_dir)

# text lowercase
def text_lowercase(text):
    return text.lower()

# remove punctuation
def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)
    
# remove whitespace from text
def remove_whitespace(text):
    return  " ".join(text.split())

# this function returns a list of tokenized and stemmed words of any text
def get_tokenized_list(doc_text):
    tokens = nltk.word_tokenize(doc_text)
    return tokens

# This function will performing stemming on tokenized words
def word_stemmer(token_list):
    ps = nltk.stem.PorterStemmer()
    stemmed = []
    for words in token_list:
        stemmed.append(ps.stem(words))
    return stemmed


# Function to remove stopwords from tokenized word list
def remove_stopwords(doc_text):
    cleaned_text = []
    for words in doc_text:
        if words not in stop_words:
            cleaned_text.append(words)
    return cleaned_text

def preprocessing_text(current_path: str):
    
    """Preprocess text by some techniques: lowercase, remove punctuation, remove white space
        
        ---------------
        Args: 
        @current_path current path of caption txt files
        
        ---------------
        Return:
        @id_corpus: list of id 
        @corpus: list of corpus
    
    """
    
    corpus = []
    id_corpus = []
    for file_name in sorted(os.listdir(current_path)):
        """ We don't use utf-8. Use cp1252 instead.
        https://stackoverflow.com/questions/46000191/utf-8-codec-cant-decode-byte-0x92-in-position-18-invalid-start-byte

        If you want to use utf-8, add try-catch exception to avoid decode error
        """
        lines = open(os.path.join(current_path, file_name), 'r', encoding='cp1252')
        temp_str = ''
        try:
            for line in lines:
                if line[-1] == '\n': # if last character is endline character
                    temp_str += line[:-1] + ' ' # delete endline character
                else:
                    temp_str += line
            temp_str = text_lowercase(temp_str)
            temp_str = remove_punctuation(temp_str)
            temp_str = remove_whitespace(temp_str)
            corpus.append(temp_str)
            id_corpus.append(str(file_name[:-4]))
            lines.close()
        except:
            pass
        
    return (corpus, id_corpus)

corpus, id_corpus = preprocessing_text(LABEL_DIR)


vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)

noisy_words = vectorizer.get_feature_names_out()
noisy_words = noisy_words[:350].tolist()

#Vector Space representation
vector = X
df1 = pd.DataFrame(vector.toarray(), columns=vectorizer.get_feature_names_out())
stop_words = set(stopwords.words('english') + noisy_words)

#Check for single document
tokens = get_tokenized_list(corpus[1])
doc_text = remove_stopwords(tokens)
doc_text = word_stemmer(doc_text)
doc_ = ' '.join(doc_text)


cleaned_corpus = []
for doc in corpus:
    tokens = get_tokenized_list(doc)
    doc_text = remove_stopwords(tokens)
    doc_text  = word_stemmer(doc_text)
    doc_text = ' '.join(doc_text)
    cleaned_corpus.append(doc_text)


vectorizerX = TfidfVectorizer()
vectorizerX.fit(cleaned_corpus)
doc_vector = vectorizerX.transform(cleaned_corpus)

df1 = pd.DataFrame(doc_vector.toarray(), columns=vectorizerX.get_feature_names_out())


def text_query(query, k=10):

    start = time.time()

    query = text_lowercase(query)
    query = remove_punctuation(query)
    query = remove_whitespace(query)
    query = get_tokenized_list(query)
    query = remove_stopwords(query)
    q = []
    for w in word_stemmer(query):
        q.append(w)
    q = ' '.join(q)
    q_stemmed = q
    query_vector = vectorizerX.transform([q])

    # calculate cosine similarities
    cosineSimilarities = cosine_similarity(doc_vector,query_vector).flatten()

    related_docs_indices = cosineSimilarities.argsort()[:-(k+1):-1]

    stop = time.time()
    running_time = stop - start
    print(running_time)
    fig = plt.figure(figsize=(15, 6))
    fig.suptitle('Query stemmed: %s \nRunning time: %.3fs' % (q_stemmed, float(running_time)))
    for idx, id in enumerate(related_docs_indices):
        img_name = str(id_corpus[id]) + '.png'
        img = mpimg.imread(os.path.join(IMAGE_DIR, img_name))
        fig.add_subplot(2, 5, idx+1)
        plt.title("Top #{}".format(idx+1))
        plt.imshow(img)
        plt.axis('off')
    plt.show()


query = 'A monitor has a message displayed on it'
text_query(query)