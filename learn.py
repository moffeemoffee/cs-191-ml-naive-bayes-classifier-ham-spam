from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from time import gmtime, strftime, time
import os
import numpy as np
import pandas as pd

dataset_path = 'dataset/trec07p/'
csv_path = 'processed-2019-02-21-00-50-30.csv'


def time_taken(task, time_start):
    time_taken = strftime('%H:%M:%S', gmtime(time() - time_start))
    print('{} took {}'.format(task, time_taken))


def learn(path):
    main_t0 = time()

    # Read file
    t0 = time()
    df = pd.read_csv(path, header=0, index_col=0, converters={
                     'words': lambda x: ' '.join(x.strip('[]').replace('\'', '').split(', '))})
    time_taken('File reading and concatenation', t0)

    # Concatenate tokens
    # t0 = time()
    # df['text'] = df['words'].apply(' '.join)
    # time_taken('Text concatenation', t0)

    # https://stackabuse.com/the-naive-bayes-algorithm-in-python-with-scikit-learn/
    t0 = time()
    count_vect = CountVectorizer()
    counts = count_vect.fit_transform(df['words'])
    time_taken('CountVectorizer fit_transform', t0)

    # Term Frequency Inverse Document Frequency
    t0 = time()
    transformer = TfidfTransformer().fit(counts)
    counts = transformer.transform(counts)
    time_taken('TfidfTransformer fit & transform', t0)

    # Split data
    t0 = time()
    x_train, x_test, y_train, y_test = train_test_split(
        counts, df['is_spam'], test_size=0.2)
    time_taken('train_test_split', t0)

    # Generate model
    t0 = time()
    model = MultinomialNB().fit(x_train, y_train)
    time_taken('MultinomialNB fit', t0)

    time_taken('Learning process', main_t0)

    return model, x_test, y_test


def evaluate(model, x_test, y_test):
    predicted = model.predict(x_test)

    accuracy = np.mean(predicted == y_test)

    print('\nAccuracy: {}'.format(accuracy))


if __name__ == '__main__':
    model, x_test, y_test = learn(os.path.join(dataset_path, csv_path))
    evaluate(model, x_test, y_test)
