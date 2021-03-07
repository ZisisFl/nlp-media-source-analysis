import multiprocessing
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import re
from sklearn import metrics
from sklearn import utils
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from articles_preprocessing_for_word2vec import create_dictionary_of_articles, text_processing, load_stop_words
from gensim import models
from os import path
from typing import Union


def remove_stop_word(text, stop_words):
    return ' '.join([token for token in text.split() if not token in stop_words])


def evaluate_model(model_type: str, y_test: np.array, prediction: np.array, labels_dict: dict) ->None:

    model_types = {'naive_bayes': 'Multinomia NB',
                   'logistic_regression': 'Logistic Regression',
                   'convolutional': 'Convolutional',
                   'lstm': 'LSTM',
                   }
    
    if model_type not in model_types.keys():
        raise ValueError('Wrong model_type \nmodel_type should be one of: "naive_bayes", "logistic_regression", "convolutional", "lstm"')

    # metrics
    print(f'Accuracy score: {metrics.accuracy_score(y_test, prediction):.3f}')
    print(f'Recall score: {metrics.recall_score(y_test, prediction, average="macro"):.3f}')
    print(f'Precision score: {metrics.precision_score(y_test, prediction, average="macro"):.3f}')
    print(f'F1 score: {metrics.f1_score(y_test, prediction, average="macro"):.3f}')

    print(f'Classification report:\n{metrics.classification_report(y_test, prediction, target_names=list(labels_dict.keys()))}')


    # confusion matrix
    cm = metrics.confusion_matrix(y_test, prediction, labels=list(labels_dict.values()))

    df_cm = pd.DataFrame(cm, index=list(labels_dict.keys()), columns=list(labels_dict.keys()))
    plt.figure(figsize=(16,7))
    plt.title(f'{model_types[model_type]} - Confusion matrix')
    sn.set(font_scale=1)
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='.0f')

    #plt.savefig(f'{model_types[model_type]}_matrix.png')
    plt.show()


def prepare_target_articles(classification_type: str='political_bias', keep_english: bool=False, balance_classes: bool=False, remove_stop_words: bool=False) ->(pd.DataFrame, dict):

    # check classification type input
    if classification_type not in ['political_bias', 'media_source']:
        raise ValueError('Wrong classification type. It can be either "political_bias" or "media_source"')

    # load stop words
    stop_words = load_stop_words()

    # name of outlet can add bias for the classification so we remove it from article body
    tokens_to_remove = ['σκαι', 'σκαϊ', 'σκάι', 'σκάϊ',
                        'εφσυν', 'ta nea', 
                        'τα νέα', 'το πρώτο θέμα',
                        'καθημερινή', 'το βήμα', 
                        'το βημα', 'καθημερινη'
                        'to vima', 'efsyn', 'skai',
                        'tovima', 'ethnos', 'tanea',
                        'protothema', 'kathimerini',
                        'kontra', 'το έθνος', 'το εθνος']

    # media sources articles is a dictionary where keys are media source names 
    # and values are dataframe of all related articles
    media_sources_articles = create_dictionary_of_articles()

    # dictionary mapping source names and class value
    source_labels_dict = {}

    for index, media_source in enumerate(media_sources_articles.keys()):
        source_labels_dict[media_source] = index

    # dictionary mapping political bias label to class value
    bias_labels_dict = {'left': 0, 'right': 1}

    # init df of all articles
    merged_df = pd.DataFrame(None)

    for media_source, articles_df in media_sources_articles.items():

        # create source label
        articles_df['source_label'] = source_labels_dict[media_source]

        # concatenate articles
        merged_df = pd.concat([merged_df, articles_df], axis=0)

    # drop rows that do not have article body
    merged_df = merged_df.dropna(subset=['main_text'])

    # filter articles on maximum overlap period
    min_date, max_date = get_maximum_period_overlap(merged_df)
    merged_df = merged_df[(merged_df['upload_time'] >= min_date) & (merged_df['upload_time'] <= max_date)]

    # keep important columns
    merged_df = merged_df[['main_text', 'source_label', 'title', 'subtitle', 'tags', 'upload_time']]

    # in case of political bias classification add the respective labels
    if classification_type == 'political_bias':
        # define political bias labels 2 left and 2 right media sources
        merged_df.loc[merged_df['source_label'] == source_labels_dict['kontra'], 'political_bias_label'] = bias_labels_dict['left']
        merged_df.loc[merged_df['source_label'] == source_labels_dict['efsyn'], 'political_bias_label'] = bias_labels_dict['left']

        merged_df.loc[merged_df['source_label'] == source_labels_dict['skai'], 'political_bias_label'] = bias_labels_dict['right']
        merged_df.loc[merged_df['source_label'] == source_labels_dict['kathimerini'], 'political_bias_label'] = bias_labels_dict['right']
                
        # remove articles that do not contain political bias label
        merged_df = merged_df.dropna(subset=['political_bias_label'])

        labels_dict = bias_labels_dict
    else:
        labels_dict = source_labels_dict

    # create processed article column
    merged_df['processed_main_text'] = merged_df['main_text'].str.lower()

    # remove tokens that refer to outlet source
    for token in tokens_to_remove:
        merged_df['processed_main_text'] = merged_df['processed_main_text'].str.replace(token, ' ')
    
    # either to keep or not tokens of written with latin characters
    if keep_english:
        # remove html 4 characters
        merged_df['processed_main_text'] = merged_df.apply(lambda x: re.sub(r'&[a-zA-Z0-9]+;', ' ', x['processed_main_text']), axis=1)
        merged_df['processed_main_text'] = merged_df.apply(lambda x: re.sub('[^Α-ΩΆΈΌΊΏΉΎΫΪ́α-ωάέόίώήύϊΐϋΰA-Za-z]', ' ', x['processed_main_text']), axis=1)
        
    else:
        merged_df['processed_main_text'] = merged_df.apply(lambda x: re.sub('[^Α-ΩΆΈΌΊΏΉΎΫΪ́α-ωάέόίώήύϊΐϋΰ]', ' ', x['processed_main_text']), axis=1)

    # sample documents in order to have balanced classes
    if balance_classes:
        # get number of articles of the inferior class
        n_articles = merged_df.groupby(['source_label']).count().min()['processed_main_text']

        # take randomly equal number of documents per class
        merged_df = merged_df.groupby(['source_label']).sample(n=n_articles)

    # choose to remove stopwords or not
    if remove_stop_words:
        merged_df['processed_main_text'] = merged_df.apply(lambda x: remove_stop_word(x['processed_main_text'], stop_words), axis=1)


    return merged_df, labels_dict


def encode_articles(encoding_type: str, X_train: Union[np.array, pd.DataFrame], X_test: Union[np.array, pd.DataFrame], 
                    y_train=None, y_test=None):

    if encoding_type == 'tf_idf':
        vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2))

        X_train = vectorizer.fit_transform(X_train)
        X_test = vectorizer.transform(X_test)

        return X_train, X_test

    elif encoding_type == 'bag_of_words':
        vectorizer = CountVectorizer(max_features=10000, ngram_range=(1,2))

        X_train = vectorizer.fit_transform(X_train)
        X_test = vectorizer.transform(X_test)

        return X_train, X_test
    
    elif encoding_type == 'word2vec': 
        path_to_embeddings = path.join('articles', 'processed','trained_embeddings')

        model = models.KeyedVectors.load_word2vec_format(path.join(path_to_embeddings, 'all100_final.bin'), binary=True)
        #model = models.KeyedVectors.load_word2vec_format(path.join('articles', 'processed','trained_embeddings', 'cc_el_300.vec'), binary=False)# 0.291
        stop_words = load_stop_words()
        
        X_train = X_train.apply(lambda x: calc_mean_vector(model, prepare_text_for_averaging(x)))
        X_train = np.array(X_train.values.tolist()) 

        X_test = X_test.apply(lambda x: calc_mean_vector(model, prepare_text_for_averaging(x)))
        X_test = np.array(X_test.values.tolist())

        return X_train, X_test
    
    elif encoding_type == 'doc2vec':
        # load stop words
        stop_words = load_stop_words()

        # create TaggedDocument representations for train and test articles (list of article tokens along with label)
        train_tagged = []
        for article, label in zip(X_train, y_train):
            train_tagged.append(models.doc2vec.TaggedDocument(words=[w for w in article.split() if w not in stop_words and len(w)>=2], tags=[label]))

        test_tagged = []
        for article, label in zip(X_test, y_test):
            test_tagged.append(models.doc2vec.TaggedDocument(words=[w for w in article.split() if w not in stop_words and len(w)>=2], tags=[label]))
        
    else:
        raise ValueError('encoding_type should be one of "tf_idf", "bag_of_words", "word2vec", "doc2vec"')

        # get number of cores to parallelize training
        cores = multiprocessing.cpu_count()

        # dm = 0 -> distributed bag of words
        model_dbow = models.Doc2Vec(dm=0, 
                                    vector_size=100, 
                                    negative=5, 
                                    hs=0, 
                                    min_count=10, 
                                    workers=cores)
        
        # build vocabulary based on train documents
        model_dbow.build_vocab(train_tagged)

        # train model
        model_dbow.train(utils.shuffle(train_tagged), total_examples=len(train_tagged), epochs=10)
        
        # final representations
        X_train = [model_dbow.infer_vector(article.words, steps=20) for article in train_tagged]
        X_test = [model_dbow.infer_vector(article.words, steps=20) for article in test_tagged]

        return X_train, X_test

    
def prepare_text_for_averaging(article: str) ->list:
    processed_article = text_processing(article)

    processed_article = processed_article.lower()

    stop_words = load_stop_words()

    split_to_tokens = processed_article.split()
    split_to_tokens = [w for w in split_to_tokens if w not in stop_words and len(w)>=2]

    return split_to_tokens


def calc_mean_vector(word2vec_model, words):
    # remove out-of-vocabulary words
    words = [word for word in words if word in word2vec_model.vocab]

    if len(words) >= 1:
        return np.mean(word2vec_model[words], axis=0)
    else:
        return np.zeros(word2vec_model.vector_size,)


def get_maximum_period_overlap(merged_df: pd.DataFrame) ->(pd.Timestamp, pd.Timestamp):

    min_date = merged_df.groupby(['source_label'])['upload_time'].min().max() 
    max_date = merged_df.groupby(['source_label'])['upload_time'].max().min() 

    return min_date, max_date
