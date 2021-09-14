# COVID-19 Topic Modeling

Results and a rough situation can be found [here](https://jesuzsh.github.io/covid-tweets-lda/).

Using a [dataset of tweets](https://github.com/echen102/covid-19-tweetids)
associated with COVID-19, Latent Dirichlet Allocation [LDA](https://radimrehurek.com/gensim_3.8.3/models/ldamodel.html)
is used to find the most common topics for each month of the dataset. Four
unique models were trained representing the months January, February, March, 
and April.

### The models

With the naming scheme is _YY-MM\_model_ along with various metafiles that
follow similar naming schemes. The _tmp/_ directory contains various files used
to incrementally components needed for training like the bigram model and word
dictionary. 

### Running the program

Download dependencies in _requirements.txt_.

The major component needed to train and create models is _covid_tweets.db_ a
SQLite3 database that can be downloaded [here](https://drive.google.com/open?id=1AmQ9ydTWMns9AgWGXDqt6iH0yOlnV48Z)(1 Gb).
Now that you have the needed data the command to train a model would be:

    python model\_magic.py <YY-MM> train

A specific model can be analyzed with the following command:

    python model\_magic.py <YY-MM> analyze
