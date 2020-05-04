import sqlite3
import json
import gzip
from pprint import pprint
from gensim.models.phrases import Phrases, Phraser
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from gensim.test.utils import datapath


class TweetLDA:
    def __init__(self, date):
        self.date = date
        self.documents = []
        self.tweet_ids = []

        self.bigram = None
        self.b_min = 90

        self.dictionary = None
        self.corpus = None
        
        # Training parameters
        self.num_topics = 6
        self.chunksize = 60000
        self.passes = 20
        self.iterations = 400
        self.eval_every = None

        self.model = None


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
            SELECT tweet_id, tokenized_tweet
            FROM token_tweets
            WHERE date = ?
            AND in_model = 0
            LIMIT 50000'''

        cursor.execute(query, (self.date,))
        results = cursor.fetchall()

        cnxn.close()

        if len(results) == 0:
            raise ValueError

        for tweet_id, tt in results:
            self.documents.append(tt.split(" "))
            self.tweet_ids.append(tweet_id)

        for i in range(len(self.documents)):
            for token in self.bigram[self.documents[i]]:
                if '_' in token:
                    self.documents[i].append(token)

        print("Documents have been prepared.")


    def update_documents(self):
        '''
        Flag that documents have been added to the LDA model

        :update: [token_tweets]
        '''
        print("Updated relevant documents in [token_tweets]")
        cnxn = sqlite3.connect("covid_tweets.db")
        cursor = cnxn.cursor()

        update_query = '''
            UPDATE token_tweets
            SET in_model = 1
            WHERE tweet_id IN (%s)'''

        cursor.execute(update_query % ','.join('?'*len(self.tweet_ids)), self.tweet_ids)
        cnxn.commit()
        cnxn.close()


    def generate_dictionary(self):
        '''
        Create a dictionary representation of the documents, filtering extremes.

        :update: self.dictionary, gensim Dictionary object
        '''
        print("Generating dictionary.")
        cnxn = sqlite3.connect("covid_tweets.db")
        cursor = cnxn.cursor()

        query = '''
            SELECT tokenized_tweet
            FROM token_tweets
            WHERE date = ?'''

        cursor.execute(query, (self.date,))
        results = cursor.fetchall()

        cnxn.close()

        self.documents = [tt.split(" ") for tt, in results]

        for i in range(len(self.documents)):
            for token in self.bigram[self.documents[i]]:
                if '_' in token:
                    self.documents[i].append(token)

        self.dictionary = Dictionary(self.documents)
        self.dictionary.filter_extremes(no_below=30, no_above=0.50)

        self.dictionary.save(f"./tmp/{self.date}_dictionary.pkl")
        print("Dictionary has been saved.")


    def load_dictionary(self):
        '''
        Load a dictionary of the associated documents.

        :update: self.corpus, list of Bag-of-Word documents
        '''
        self.dictionary = Dictionary()
        self.dictionary = self.dictionary.load(f"./tmp/{self.date}_dictionary.pkl")

        print("Dictionary loaded.")

    
    def generate_corpus(self):
        '''
        Create a Bag-of-Words representation corpora. Ready to be trained.

        :update: self.corpus, list of Bag-of-Word documents
        '''
        self.corpus = [self.dictionary.doc2bow(d) for d in self.documents]


    def generate_model(self):
        '''
        Utilizting the python Gensim library and the prepared corpus, create a
        trained LDA model.

        :update: self.model, LdaModel object
        '''
        temp = self.dictionary[0]
        id2word = self.dictionary.id2token

        print("Model generation is beginning.")

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

        print("Model generated.")

        temp_file = datapath(f"{self.date}_model")
        print(temp_file)
        self.model.save(temp_file)
        print("Model has been saved.")

        self.update_documents()

        pprint(self.model.top_topics(self.corpus))


    def load_model(self):
        '''
        Load a pre-trained model to be analyze or updated

        :update: self.model, LdaModel object
        '''
        temp_file = datapath(f"{self.date}_model")
        print(temp_file)
        self.model = LdaModel.load(temp_file)


    def update_model(self):
        '''
        Update the pre-existing model with a new corpus

        :update: self.documents
        :update: self.model
        :udpate: self.corpus
        '''
        self.prepare_documents()
        self.generate_corpus()

        print(f"{self.date}_model is being updated.")
        self.model.update(self.corpus, chunksize=self.chunksize)

        temp_file = datapath(f"{self.date}_model")
        print(temp_file)
        self.model.save(temp_file)
        print("Model has been saved.")

        pprint(self.model.top_topics(self.corpus))

        self.update_documents()


    def analyze_model(self):
        '''
        Examine the top topics of the model.
        '''

        cnxn = sqlite3.connect("covid_tweets.db")
        cursor = cnxn.cursor()

        query = '''
            SELECT tokenized_tweet
            FROM token_tweets
            WHERE date = ?
            AND in_model = 1'''

        cursor.execute(query, (self.date,))
        results = cursor.fetchall()

        cnxn.close()

        print(len(results), "documents are in the model.")

        for tt, in results:
            self.documents.append(tt.split(" "))

        for i in range(len(self.documents)):
            for token in self.bigram[self.documents[i]]:
                if '_' in token:
                    self.documents[i].append(token)

        self.generate_corpus()
 
        self.top_topics = self.model.top_topics(self.corpus)
        pprint(self.top_topics)
        self.save_top_topics()


    def output_topics_json(self, values):
        '''
        Output json from a list of tuples
        '''
        to_json = []
        for date, topic_num, word, probability in values:
            to_json.append({"date": date, "topic_num": topic_num, "word": word, "probability": str(probability)})

        with open(f"./tmp/{self.date}_topics.json", "w") as outfile:
            json.dump(to_json, outfile)


    def save_top_topics(self):
        '''
        Given the top topics, save them to a .json file
        '''
        to_save = [] 
        for i, topic in enumerate(self.top_topics, 1):
            for probability, word in topic[0]:
                to_save.append((self.date, i, word, probability)) 

        self.output_topics_json(to_save)
