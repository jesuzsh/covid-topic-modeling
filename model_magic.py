from tweet_lda import TweetLDA
from pprint import pprint


if __name__ == "__main__":
    tlda = TweetLDA("2020-01")

    tlda.compute_bigrams()

    '''
    tlda.collect_documents()
    tlda.preprocess_documents()

    model, corpus = tlda.generate_model()

    pprint(model.top_topics(corpus))
    '''
