import codecs
import os
import nltk
import re
import pickle
import pandas as pd
from os import path, listdir
from glob import glob
from pathlib import Path
from tqdm import tqdm


def load_stop_words():
    
    # load greek stop words from file (taken from spaCy)
    stop_words_file = open('stop_words.txt', 'r', encoding='utf-8-sig')
    stop_words = []
    for line in stop_words_file.readlines():
        stop_words.append(line[:-1])
    
    return stop_words


def text_processing(raw_text):
    # remove HTML4 characters https://www.w3schools.com/charsets/ref_html_entities_4.asp
    #raw_text = re.sub(r'&[a-zA-Z0-9]+;', ' ', raw_text)

    # remove punctuation to prevent splitting acronyms
    # example ΕΛ.ΑΣ would be ΕΛ ΑΣ or Ε.Ε would be Ε Ε
    #raw_text = re.sub(r'[^\w\s]', '', raw_text)

    # keep tokens containing the following characters
    processed_text = re.sub('[^Α-ΩΆΈΌΊΏΉΎΫΪ́α-ωάέόίώήύϊΐϋΰ]', ' ', raw_text)

    return processed_text


def create_dictionary_of_articles():
    # define original article destination
    original_articles_destination = Path('articles') / 'original'
    
    # create list of article source paths
    media_sources_paths = original_articles_destination.glob('*')

    # create a dictionary of dataframes per source
    media_sources_articles = {}

    for media_source in media_sources_paths:
        # extract media source name from path
        media_source_name = media_source.name.split('_articles')[0]

        # list files in folder
        csv_files = media_source.glob('*.csv')
        
        df_list = [] 

        for csv_file in csv_files:
            media_source_df = pd.read_csv(csv_file)

            # transform time column to pandas datetime
            media_source_df['upload_time'] = pd.to_datetime(media_source_df['upload_time'], utc=True)
            media_source_df['update_time'] = pd.to_datetime(media_source_df['update_time'], utc=True)
            df_list.append(media_source_df)

        merged_articles_df = pd.concat(df_list, axis=0)
            
        media_sources_articles[media_source_name] = merged_articles_df
    
    return media_sources_articles


if __name__ == "__main__":
    # load nltk greek tokenizer
    tokenizer = nltk.data.load('tokenizers/punkt/greek.pickle')

    # greek stop words from spaCy
    stop_words = load_stop_words()
    
    # a dictionary containing a dataframe of articles for each source
    media_sources_articles = create_dictionary_of_articles()

    for media_source_name, articles_df in media_sources_articles.items():
        total_articles = len(articles_df.index)
        print(media_source_name)

        # create pickle files containing all sentences of articles from each media source
        media_source_sentences_pkl = open(path.join('articles', 'processed', 'word2vec_pickles', '{}.pkl'.format(media_source_name)), 'wb')

        for article_index, article in tqdm(articles_df.iterrows(), total=total_articles):
            
            if isinstance(article['main_text'], str):

                # using nltk greek tokenizer tokenize article to sentences
                raw_sentences = tokenizer.tokenize(article['main_text'])

                for raw in raw_sentences:
                    # process sentence 
                    clean = text_processing(raw)
                    
                    # split sentence to tokens list
                    tokens = clean.split()

                    # init filtered words list
                    filtered_tokens = []

                    for token in tokens:
                        token = token.lower()
                        # keep only tokens of 2 characters or more
                        if len(token) >= 2:
                            # remove stop words
                            if token not in stop_words:
                                filtered_tokens.append(token)

                    #print(filtered_tokens)

                    # discard lists that are empty
                    if len(filtered_tokens) > 0:
                        pickle.dump(filtered_tokens, media_source_sentences_pkl)
