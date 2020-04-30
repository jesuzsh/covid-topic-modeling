import nltk
nltk.download('wordnet')
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models import Phrases
from gensim.corpora import Dictionary


import gzip
import json


class CovidTweets:
    def __init__(self):
        self.file = "./data/coronavirus-tweet-id-2020-01-21-22.jsonl.gz"
        self.data = []
        self.documents = []

        self.dictionary = None
        self.corpus = None
        
        # Training parameters
        num_topics = 10
        chunksize = 1000
        passes = 20
        iterations = 400
        eval_every = None

           
    def extract_tweets(self):
        '''
        Obtain a list of tweets from a filename

        :update: self.data, list of tweets from file as python dictionaries
        '''
        with gzip.open(self.file) as f:
            tweets_jsonl = f.read().decode("utf-8")
            tweets_list = tweets_jsonl.split('\n')

            for jline in tweets_list:
                if jline != "":
                    current_tweet = json.loads(jline)
                    if current_tweet['lang'] == 'en':
                        self.data.append(current_tweet)


    def collect_documents(self):
        '''
        Generate a list of Strings from list of tweet Dictionaries

        :update: self.documents, list of Strings
        '''
        self.documents = [tweet['full_text'] for tweet in self.data]


    def tokenize_documents(self):
        '''
        Tokenize the documents

        :update: self.documents, list of tokenized Strings
        '''
        tokenizer = RegexpTokenizer(r'\w+')
        for i in range(len(self.documents)):
            self.documents[i] = self.documents[i].lower()
            self.documents[i] = tokenizer.tokenize(self.documents[i])

        self.documents = [[token for token in d if not token.isnumeric()] for d in self.documents]
        self.documents = [[token for token in d if len(token) > 1] for d in self.documents]


    def lemmatize_documents(self):
        '''
        Lemmatize the documents (e.g. its => it)

        :update: self.documents, lists of lemmatized Strings
        '''
        lemmatizer = WordNetLemmatizer()
        self.documents = [[lemmatizer.lemmatize(token) for token in d] for d in self.documents]


    def compute_bigrams(self):
        '''
        Find and save bigrams located in the documents

        :param docs: List, tokenized and lemmatized documents 
        :return: None
        '''
        bigram = Phrases(self.documents, min_count=20)

        for i in range(len(self.documents)):
            for token in bigram[self.documents[i]]:
                if '_' in token:
                    print("Found bigram:", token)
                    self.documents[i].append(token)


    def generate_dictionary(self):
        '''
        Create a dictionary representation of the documents, filtering extremes
        Additionally, create a ready to be trained corpus

        :update: self.dictionary, gensim Dictionary object
        :update: self.corpus, list of Bag-of-Word documents
        '''
        self.dictionary = Dictionary(self.documents)
        self.dictionary.filter_extremes(no_below=3, no_above=0.75)

        self.corpus = [self.dictionary.doc2bow(d) for d in self.documents]


    def preprocess_documents(self):
        '''
        Tokenize, lemmatize, and find bigrams of the documents

        :update: self.documents, list of Strings
        :update: self.dictionary, gensim Dictionary object
        :update: self.corpus, list of Bag-of-Word documents to be trained
        '''
        self.tokenize_documents()
        self.lemmatize_documents()
        self.compute_bigrams()

        self.generate_dictionary()
