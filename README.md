# Covid-19 Topic Modeling

Using a [dataset of tweets][https://github.com/echen102/covid-19-tweetids] associated with COVID-19, Latent Dirichlet Allocation (LDA) is used to find the most common topics for each month of the dataset. Four unique models were trained representing the months January, February, March, and April.

### The models

With the naming scheme is *YY-MM\_model* along with various metafiles that follow similar naming schemes. The *tmp/* directory contains various files used to incrementally components needed for training like the bigram model and word dictionary. 


### Running the program

Download dependencies in *requirements.txt*

The major component needed to train and create models is *covid_tweets.db* a SQLite3 database that can be downloaded [here][](1 Gb). Now that you have the needed data the command to train a model would be:

'''
python model\_magic.py <YY-MM> train
'''

A specific model can be analyzed with the following command:

'''
python model\_magic.py <YY-MM> analyze
'''
