from tweet_lda import TweetLDA
from pprint import pprint

if __name__ == "__main__":
    tlda = TweetLDA("2020-04")

    try:
        tlda.load_bigram()
    except Exception as e:
        print(e)
        tlda.compute_bigram()
        tlda.load_bigram()

    try:
        tlda.load_dictionary()
    except Exception as e:
        print(e)
        tlda.generate_dictionary()
        tlda.load_dictionary()


    '''
    tlda.load_model()
    tlda.analyze_model()
    '''
    try:
        tlda.load_model()
        tlda.update_model()
    except ValueError:
        print("The model is complete.")
    except Exception as e:
        print(e)
        tlda.prepare_documents()
        tlda.generate_corpus()
        tlda.generate_model()


    #tlda.analyze_model()
