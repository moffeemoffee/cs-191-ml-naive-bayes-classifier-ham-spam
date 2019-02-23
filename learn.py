from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from time import gmtime, strftime, time
import os
import numpy as np
import pandas as pd

dataset_path = 'dataset/trec07p/'
csv_path = 'processed-2019-02-21-00-50-30.csv'


def learn(x_train, y_train):
    gen_vocab_model = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB(alpha=1.0e-10))
    ])
    gen_vocab_model.fit(x_train, y_train)

    gen_vocab_laplace_model = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB(alpha=1))
    ])
    gen_vocab_laplace_model.fit(x_train, y_train)

    return gen_vocab_model, gen_vocab_laplace_model


def evaluate(model, model_name, x_test, y_test, y_names):
    print(model_name)
    predicted = model.predict(x_test)
    print('Accuracy: {:.05%}'.format(np.mean(predicted == y_test)))
    print(metrics.classification_report(
        y_test, predicted, target_names=y_names))


if __name__ == '__main__':
    # Read file
    df = pd.read_csv(os.path.join(dataset_path, csv_path),
                     header=0, index_col=0)

    # Split data
    target_names = ['ham', 'spam']
    x_train, x_test, y_train, y_test = train_test_split(
        df['words'], df['is_spam'], test_size=0.2, random_state=191)

    # Train models
    gvoc_model, gvoclap_model = learn(x_train, y_train)

    # Evaluate models
    evaluate(gvoc_model, 'General Vocabulary (No laplace smoothing; alpha=0.0000000001≈0)',
             x_test, y_test, target_names)
    evaluate(gvoclap_model, 'General Vocabulary (With laplace smoothing; alpha=1)',
             x_test, y_test, target_names)
