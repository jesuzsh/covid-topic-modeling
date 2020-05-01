from tweet_lda import TweetLDA
from pprint import pprint

if __name__ == "__main__":
    tlda = TweetLDA("2020-01")

    try:
        tlda.load_bigram()
    except Exception as e:
        tlda.compute_bigram()
        tlda.load_bigram()

    try:
        tlda.load_documents()
    except Exception as e:
        tlda.prepare_documents()

    tlda.generate_dictionary()

    model, corpus = tlda.generate_model()

    pprint(model.top_topics(corpus))
