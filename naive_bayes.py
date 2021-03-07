import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
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

    # model
    # parameters to search
    parameters = {'alpha': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}
    naive_bayes = MultinomialNB()

    clf = GridSearchCV(naive_bayes, parameters)
    clf.fit(X_train, y_train)

    # use best estimator to make the prediction
    clf = clf.best_estimator_

    # predict
    prediction = clf.predict(X_test)

    # add the information on the data to save
    test_articles_df['pred_label'] = prediction.astype(int)
    test_articles_df.to_csv(f'articles/results/naive_bayes_{classfication_type}_results.csv', index=False)

    # assess performance
    evaluate_model('naive_bayes', y_test, prediction, labels_dict)
