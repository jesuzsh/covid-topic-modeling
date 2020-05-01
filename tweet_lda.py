import sqlite3
from gensim.models import Phrases
from gensim.corpora import Dictionary
from gensim.models import LdaModel



class TweetLDA:
    def __init__(self):
        self.documents = []

        self.dictionary = None
        self.corpus = None
        
        # Training parameters
        self.num_topics = 12
        self.chunksize = 1000
        self.passes = 20
        self.iterations = 400
        self.eval_every = None

        self.model = None


    def collect_documents(self):
        '''
        Generate a list of Strings from list of tweet SQL entries

        :update: self.documents, list of Strings
        '''
        cnxn = sqlite3.connect("covid_tweets.db")
        cursor = cnxn.cursor()

        self.documents = [t for t, in cursor.execute("SELECT tweet FROM tweets")]

        cnxn.close()


    def compute_bigrams(self):
        '''
        Find and save bigrams located in the documents

        :update: self.documents, list of Strings + bigrams
        '''
        bigram = Phrases(self.documents, min_count=90)

        for i in range(len(self.documents)):
            for token in bigram[self.documents[i]]:
                if '_' in token:
                    self.documents[i].append(token)


    def generate_dictionary(self):
        '''
        Create a dictionary representation of the documents, filtering extremes
        Additionally, create a ready to be trained corpus

        :update: self.dictionary, gensim Dictionary object
        :update: self.corpus, list of Bag-of-Word documents
        '''
        self.dictionary = Dictionary(self.documents)
        self.dictionary.filter_extremes(no_below=30, no_above=0.50)

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
        print("Documents have been tokenized.")

        self.compute_bigrams()
        print("Bigrams have been computed.")


        self.generate_dictionary()


    def generate_model(self):
        '''
        Utilizting the python Gensim library and the prepared corpus, create a
        trained LDA model.

        :update: self.model, LdaModel object
        '''
        temp = self.dictionary[0]
        id2word = self.dictionary.id2token

        self.model = LdaModel(
            corpus=self.corpus,
            id2word=id2word,
            chunksize=self.chunksize,
            alpha='auto',
            eta='auto',
            iterations=self.iterations,
            num_topics=self.num_topics,
            passes=self.passes,
            eval_every=self.eval_every
        )


        return self.model, self.corpus

