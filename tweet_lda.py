import sqlite3
from gensim.models.phrases import Phrases, Phraser
from gensim.corpora import Dictionary
from gensim.models import LdaModel



class TweetLDA:
    def __init__(self, date):
        self.date = date
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


    """
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
    """


    def compute_bigrams(self):
        '''
        Find and save bigrams living among the tweets

        :update: [covid_tweets].[token_tweets]
        '''
        cnxn = sqlite3.connect("covid_tweets.db")
        cursor = cnxn.cursor()

        count_query = '''
            SELECT count(tweet_id)
            FROM token_tweets
            WHERE date = ?'''

        cursor.execute(count_query, (self.date,))
        num_tweets = cursor.fetchone()[0]
        print(self.date, num_tweets, "to have bigram computed.")

        update_query = '''
            UPDATE token_tweets
            SET tokenized_tweet = ?, has_bigram = 1
            WHERE tweet_id = ?'''

        query = '''
            SELECT tweet_id, tokenized_tweet
            FROM token_tweets
            WHERE date = ?'''

        cursor.execute(query, (self.date,))
        results = cursor.fetchall()

        retokenized_tweets = []
        for tweet_id, tokenized_tweet in results:
            tweet_tokens = tokenized_tweet.split(" ")
            retokenized_tweets.append(tweet_tokens)

        m_count = 90
        phrases = Phrases(retokenized_tweets, min_count=m_count)
        bigram = Phraser(phrases)

        bigram.save(f"./tmp/{self.date}_bigram_model_{m_count}.pkl")
        print("Bigram computed.")


        '''
        for i in range(num_tweets):
            for token in bigram[retokenized_tweets[i]]:
                if '_' in token:
                    retokenized_tweets[i].append(token)

            updated_tweet = " ".join(retokenized_tweet[i])
            cursor.execute(insert_query, (updated_tweet, results[i][0]))

            
        cnxn.commit()
        '''
        cnxn.close() 


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

