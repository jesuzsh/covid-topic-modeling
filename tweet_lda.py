import sqlite3
import json
import gzip
from gensim.models.phrases import Phrases, Phraser
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from gensim.test.utils import datapath



class TweetLDA:
    def __init__(self, date):
        self.date = date
        self.documents = []

        self.bigram = None
        self.b_min = 90

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


    def compute_bigram(self):
        '''
        Find and save bigrams living among the tweets

        :update: [covid_tweets].[token_tweets]
        '''
        print("Computing bigram.")
        cnxn = sqlite3.connect("covid_tweets.db")
        cursor = cnxn.cursor()

        count_query = '''
            SELECT count(tweet_id)
            FROM token_tweets
            WHERE date = ?'''

        cursor.execute(count_query, (self.date,))
        num_tweets = cursor.fetchone()[0]
        print(self.date, num_tweets, "to have bigram computed.")

        query = '''
            SELECT tweet_id, tokenized_tweet
            FROM token_tweets
            WHERE date = ?'''

        cursor.execute(query, (self.date,))
        results = cursor.fetchall()

        cnxn.close() 

        retokenized_tweets = []
        for tweet_id, tokenized_tweet in results:
            tweet_tokens = tokenized_tweet.split(" ")
            retokenized_tweets.append(tweet_tokens)

        phrases = Phrases(retokenized_tweets, min_count=self.b_min)
        bigram = Phraser(phrases)

        bigram.save(f"./tmp/{self.date}_bigram_model_{self.b_min}.pkl")
        print("Bigram computed.")


    def load_bigram(self):
        '''
        Search for and load a pre-existing bigrams file

        :update: self.bigram
        '''
        self.bigram = Phraser.load(f"./tmp/{self.date}_bigram_model_{self.b_min}.pkl")

        print("Bigram loaded.")


    def prepare_documents(self):
        '''
        Integrate bigrams into the documents so they can be used for the model

        :update: self.documents
        '''
        print("Preparing documents.")
        cnxn = sqlite3.connect("covid_tweets.db")
        cursor = cnxn.cursor()

        query = '''
            SELECT tokenized_tweet
            FROM token_tweets
            WHERE date = ?'''

        cursor.execute(query, (self.date,))
        self.documents = cursor.fetchall()

        cnxn.close()

        self.documents = [tt.split(" ") for tt, in self.documents]

        for i in range(len(self.documents)):
            for token in self.bigram[self.documents[i]]:
                if '_' in token:
                    self.documents[i].append(token)
        
        with gzip.open(f"./tmp/{self.date}_prepared_documents.json", 'wt', encoding="ascii") as zipfile:
            json.dump(self.documents, zipfile)
        print("Documents have been prepared.")


    def load_documents(self):
        '''
        Read in ready-to-use documents list from json file

        :update: self.documents
        '''
        with gzip.open(f"./tmp/{self.date}_prepared_documents.json.gz") as f:
            self.documents = json.load(f)

        print("Prepared documents have been loaded.")


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

        temp_file = datapath(f"./tmp/{self.date}_model")
        self.model.save(temp_file)

        return self.model, self.corpus

