import os
import gzip
import json
import sqlite3


def create_database():
    '''
    Create a SQLite database to store the original tweets
    '''
    cnxn = sqlite3.connect('covid_tweets.db')

    cursor = cnxn.cursor()
    cursor.execute('''CREATE TABLE tweets
                          (date text, filename text, tweet text)''')
    cnxn.commit()
    cnxn.close()


def find_files(filepath):
    '''
    Get a list of all the data files in a given filepath

    :param filepath: String, the path in which to begin file collection
    :return: list of files to be processed
    '''
    files = []

    if filepath[-1] != '/':
        filepath = filepath + '/'

    for (dirpath, dirnames, filenames) in os.walk(filepath):
        for f in filenames:
            files.append(dirpath + '/' + f)

    return files


def check_if_processed(filename):
    '''
    Query the database to see if the filename exists

    :param filename: String, name of file to be checked
    :return: Boolean, processed or not
    '''
    cnxn = sqlite3.connect('covid_tweets.db')
    cursor = cnxn.cursor()
    
    f = (filename,)
    exists_query = '''
        SELECT EXISTS(
        SELECT *
        FROM tweets
        WHERE filename = ?)'''

    cursor.execute(exists_query, f)
    processed = cursor.fetchone()[0]

    cnxn.close()

    return processed
    

def extract_tweets(filepath):
    '''
    Obtain a list of tweets from a filename

    :param filepath: String, the file to be processed
    :return: List, tweets from file as python dictionaries
    '''
    filename = filepath.split('/')[-1]

    already_saved = check_if_processed(filename) 

    if already_saved:
        print(filepath, "already in database.")
        return None

    try:
        data = []

        with gzip.open(filepath) as f:
            tweets_jsonl = f.read().decode("utf-8")
            tweets_list = tweets_jsonl.split('\n')

            for jline in tweets_list:
                if jline != "":
                    current_tweet = json.loads(jline)

                    if current_tweet['lang'] == 'en':
                        data.append(current_tweet)

        return data
    except Exception as e:
        print(filepath, "was corrupted.")
        paths = filepath.split("/")
        #os.remove("../COVID-19-TweetIDs/" + paths[-2] + "/" + paths[-1])
        #os.remove(filepath)

        return None


def save_tweets(data, path_elems):
    '''
    Send the tweets to the SQLite database

    :param data: List, tweets as python dictionaries
    :param path_elems: Tuple, filename and year-month the tweet was tweeted
    '''
    filename, date = path_elems
    
    cnxn = sqlite3.connect('covid_tweets.db')
    cursor = cnxn.cursor()

    insert_query = '''
        INSERT INTO tweets (date, filename, tweet)
        VALUES (?, ?, ?)'''

    to_insert = []
    for tweet in data:
        to_insert.append((date, filename, tweet['full_text']))

    cursor.executemany(insert_query, to_insert)

    cnxn.commit()
    cnxn.close()

    print(filename, "added to the database.")



def process_files(files):
    '''
    Get tweets from files to the SQLite db

    :param files: List, the files that contain the tweets
    '''
    for filepath in files:
        data = extract_tweets(filepath)

        if data == None:
            pass
        else:
            path_elems = filepath.split('/')

            filename = path_elems[-1]
            date = path_elems[-2]

            save_tweets(data, (filename, date))


if __name__ == "__main__":
    files = find_files("./data")
    
    process_files(files)

