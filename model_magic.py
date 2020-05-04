import argparse

from tweet_lda import TweetLDA
from pprint import pprint

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LDA on Covid Tweets.")
    parser.add_argument('date', type=str)
    parser.add_argument('choice', type=str, help="'analyze' or 'train' model")

    args = parser.parse_args()

    tlda = TweetLDA(args.date)

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

    if args.choice == 'analyze':
        tlda.load_model()
        tlda.analyze_model()
    elif args.choice == 'train':
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
    else:
        print("Incorrect choice parameter.")
