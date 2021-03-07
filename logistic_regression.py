import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from gensim import models
from articles_preprocessing_for_word2vec import load_stop_words
from os import path
from tqdm import tqdm
import re
from models_helper_functions import evaluate_model, prepare_target_articles, encode_articles
from sklearn.model_selection import GridSearchCV


if __name__ == "__main__":
    
    classfication_type = 'political_bias' # political_bias/media_source

    # construct dataframe for training and testing articles
    merged_df, labels_dict = prepare_target_articles(classification_type=classfication_type, 
                                                        keep_english=False, 
                                                        balance_classes=False,
                                                        remove_stop_words=True)

    # create train test
    classif_label_dict = {'political_bias': 'political_bias_label',
                         'media_source': 'source_label',
                         }
    X_train, X_test, y_train, y_test = train_test_split(merged_df['processed_main_text'], merged_df[classif_label_dict[classfication_type]], 
                                                        test_size=0.25, random_state=42)
    test_articles_df = pd.DataFrame(data={'article': X_test, 'label': y_test.astype(int)})

    # encode articles
    X_train, X_test = encode_articles('tf_idf', X_train, X_test)
    #X_train, X_test = encode_articles('bag_of_words', X_train, X_test)
    #X_train, X_test = encode_articles('word2vec', X_train, X_test)
    #X_train, X_test = encode_articles('doc2vec', X_train, X_test, y_train, y_test)


    # model
    # parameters to search
    parameters = {'max_iter': [250, 300]}
    logistic_regression = LogisticRegression(solver='lbfgs')

    clf = GridSearchCV(logistic_regression, parameters)
    clf.fit(X_train, y_train)

    # use best estimator to make the prediction
    clf = clf.best_estimator_

    # predict
    prediction = clf.predict(X_test)

    # add the information on the data to save
    test_articles_df['pred_label'] = prediction.astype(int)
    test_articles_df.to_csv(f'articles/results/log_reg_{classfication_type}_results.csv', index=False)

    # assess performance
    evaluate_model('logistic_regression', y_test, prediction, labels_dict)
