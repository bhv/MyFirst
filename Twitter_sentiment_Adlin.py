import sys
from datetime import datetime

import pandas
import re
import numpy
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
from sklearn import cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split

def log_info(msg):
    sys.stderr.write('%s :: %s\n' % (datetime.now(), msg))
    sys.stderr.flush()
    pass

class TwitterSentiment():
    def __init__(self):
        self.punct = re.compile(r'([^A-Za-z0-9 ])')
        self.stop = set(stopwords.words("english"))

    def read_data(self, data_path):
        print("Reading file...")
        data = pandas.read_excel("%s/Tweetsdata_analyticsteam.xlsx" % data_path)
        data = data[pandas.notnull(data['Tweet'])]
        data = data[pandas.notnull(data['Human Understanding Sentiment'])]
        print(data.shape)
        #------------ Data cleaning -------
        print("Data cleaning...")

        cleaned_tweets = []
        categories = []

        for tweet in list(data['Tweet']):
            cleaned_text = " ".join(filter(lambda word: word not in self.stop, tweet.split())).lower() # Removes stopwors & converts to lower case

            wnl = WordNetLemmatizer()
            # cleaned_text = " ".join([wnl.lemmatize(i, j[0].lower()) if j[0].lower() in ['a', 'n', 'v'] else wnl.lemmatize(i) for i, j in pos_tag(word_tokenize(cleaned_text))])
            cleaned_text = " ".join([wnl.lemmatize(i) for i in cleaned_text.split()]) # Lemmatization

            cleaned_text = self.punct.sub("", cleaned_text).strip() # Removes punctuations & unicodes

            cleaned_tweets.append(cleaned_text)
            # print("%s \n CLEANED TEXT - %s" %(tweet, cleaned_text))

        for cat in list(data['Human Understanding Sentiment']):
            cleaned_cat = cat.strip().lower()
            if cleaned_cat == "negative":
                categories.append(0)
            # elif cleaned_cat == "neutral":
            #     categories.append(1)
            # elif cleaned_cat == 'positive':
            #     categories.append(2)
            elif cleaned_cat == 'neutral' or cleaned_cat == 'positive':
                categories.append(1)
            else:
                raise ValueError('Unknown value %s' % categories)
        return cleaned_tweets, categories

    def model_creation_SVC(self, tweets, categories):

        log_info('File read & cleaned. Len X: %s, Len Y: %s' % (len(tweets), len(categories)))

        categories = numpy.asarray(categories)

        from sklearn.svm import LinearSVC
        lsvc = LinearSVC(C=1, fit_intercept=True, penalty='l2', dual=False, multi_class='ovr', tol=0.001)

        from sklearn.pipeline import Pipeline
        linear_svm_pipeline = Pipeline([('TfIdf', TfIdf()), ('lsvc', lsvc)])

        classifier_name = 'TfIdf + Linear SVC'
        classifier_obj = linear_svm_pipeline

        log_info('%s :: Running cross validation' % classifier_name)
        scores = cross_validation.cross_val_score(classifier_obj, tweets, categories, cv=5, verbose=3, n_jobs=2)
        log_info("%s :: Accuracy: %0.2f (+/- %0.2f)" % (classifier_name, scores.mean(), scores.std() * 2))
        log_info("%s :: ALL Scores : %s" % (classifier_name, scores))

        log_info('Confusion matrix with 80% data:')
        X_train, X_test, y_train, y_test = train_test_split(tweets, categories, test_size=0.2)

        lsvc = LinearSVC(C=1, fit_intercept=True, penalty='l2', dual=False, multi_class='ovr', tol=0.001)
        linear_svm_pipeline = Pipeline([('TfIdf', TfIdf()), ('lsvc', lsvc)])
        linear_svm_pipeline.fit_transform(X_train, y_train)

        y_pred = linear_svm_pipeline.predict(X_test)
        from sklearn.metrics import confusion_matrix
        print(confusion_matrix(y_test, y_pred))
        from sklearn.metrics import accuracy_score
        print('Accuracy: ', accuracy_score(y_test, y_pred))

    def model_creation_XGBoost(self, tweets, categories):
        log_info('File read & cleaned. Len X: %s, Len Y: %s' % (len(tweets), len(categories)))
        categories = numpy.asarray(categories)

        from xgboost import XGBClassifier
        from sklearn.pipeline import Pipeline
        xgb_model = XGBClassifier()
        xgboost_pipeline = Pipeline([('TfIdf', TfIdf()), ('XGBoost', xgb_model)])
        classifier_name = 'TfIdf + XGBoost'
        classifier_obj = xgboost_pipeline

        log_info('%s :: Running cross validation' % classifier_name)
        scores = cross_validation.cross_val_score(classifier_obj, tweets, categories, cv=5, verbose=3, n_jobs=2)
        log_info("%s :: Accuracy: %0.2f (+/- %0.2f)" % (classifier_name, scores.mean(), scores.std() * 2))
        log_info("%s :: ALL Scores : %s" % (classifier_name, scores))

        log_info('Confusion matrix with 80% data:')
        X_train, X_test, y_train, y_test = train_test_split(tweets, categories, test_size=0.2)

        xgb_model = XGBClassifier()
        xgb_pipeline = Pipeline([('TfIdf', TfIdf()), ('XGBoost', xgb_model)])
        xgb_pipeline.fit(X_train, y_train)
        y_pred = xgb_pipeline.predict(X_test)
        from sklearn.metrics import confusion_matrix
        print(confusion_matrix(y_test, y_pred))
        from sklearn.metrics import accuracy_score
        print('Accuracy: ', accuracy_score(y_test, y_pred))

from sklearn.base import BaseEstimator, TransformerMixin
class TfIdf(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.tfidf = None

    def fit(self, X_input, y=None):
        log_info('TfIdf fit() - len X_input: %s' % len(X_input))
        self.tfidf = TfidfVectorizer(min_df=1, max_df=.9, ngram_range=(1, 3), strip_accents='unicode', norm='l2')
        self.tfidf.fit(X_input)
        return self

    def transform(self, X_input):
        log_info('TfIdf transform() - len X_input: %s' % len(X_input))
        X_output = self.tfidf.transform(X_input)
        log_info('TfIdf transform() - len X_output: %s' % str(X_output.shape))
        return X_output


if __name__ == '__main__':
    Data_path = "C:/Users/kmbl111798/PycharmProjects/Twitter_analysis/"
    sentiment = TwitterSentiment()
    cleaned_tweets, categories = sentiment.read_data(Data_path)
    # print(cleaned_tweets)
    # print(categories)
    sentiment.model_creation_SVC(cleaned_tweets,categories)
    # sentiment.model_creation_XGBoost(cleaned_tweets,categories)