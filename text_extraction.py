from covid_tweets import CovidTweets


if __name__ == "__main__":
    ct = CovidTweets()

    ct.extract_tweets()
    ct.collect_documents()
    ct.preprocess_documents()


