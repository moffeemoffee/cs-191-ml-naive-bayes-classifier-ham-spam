from bs4 import BeautifulSoup
from datetime import datetime
from mailparser import parse_from_file
from math import floor
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from time import gmtime, sleep, strftime, time
from tqdm import tqdm
import concurrent.futures
import contractions
import logging
import lxml
import multiprocessing
import numpy as np
import nltk
import os
import pandas as pd
import re
import string

exit_flag = 0

dataset_path = 'dataset/trec07p/'
index_path = 'full/index'
csv_path = 'processed-{}.csv'.format(
    datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))

punc_regex = re.compile('[%s]' % re.escape(string.punctuation))

cached_stopwords = stopwords.words('english')

lemmatizer = WordNetLemmatizer()


def preprocess(index):
    files = read_data(index)

    results = parallel_preprocess(files)
    results = after_preprocess_clean(results)
    results = pd.DataFrame(results)

    results.to_csv(os.path.join(dataset_path, csv_path))

    return results


def read_data(index):
    time_start = time()

    files = pd.read_csv(index, sep=' ', names=[
                        'is_spam', 'email_path'])
    files['is_spam'] = files['is_spam'].map({'spam': 1, 'ham': 0})
    files['words'] = ''
    # files['counts'] = ''

    time_taken = strftime('%H:%M:%S', gmtime(time() - time_start))
    print('Reading took {} for {} files'.format(time_taken, files.shape[0]))
    return files


def parallel_preprocess_func(d):
    # global total_file_size
    # global dropped_files
    # global dropped_empty
    # global dropped_exception

    # tqdm.write('{} {}'.format(str(type(d)), str(d)))
    # tqdm.write('{} {}'.format(str(type(row)), str(row)))

    # tqdm.write('===')
    # for x in d:
    #     tqdm.write('{} {}'.format(str(type(x)), str(x)))

    index = d[0]
    row = d[1]

    try:
        email_path = os.path.join(
            dataset_path, index_path, '..', row['email_path'])
        email_path = os.path.abspath(email_path)
        # total_file_size += os.stat(email_path).st_size
        email_body = preprocess_text(get_email_body_from_file(email_path))
        if not email_body:
            # tqdm.write('\nString at [{}]{} is empty'.format(index, row['email_path']))
            # dropped_files += 1
            # dropped_empty += 1
            # d.drop(index, inplace=True)
            row = None
        else:
            email_words = preprocess_text_tokenize(email_body)
            # d.at[index, 'words'] = email_words
            row['words'] = email_words
    except Exception as e:
        tqdm.write('Exception at {}'.format(row['email_path']))
        logging.exception('message')
        # dropped_files += 1
        # dropped_exception += 1
        # d.drop(index, inplace=True)
        row = None

    return row


def parallel_preprocess(df, num_processes=None):
    time_start = time()

    if num_processes is None:
        num_processes = multiprocessing.cpu_count()

    # print(df)

    # https://www.tjansson.dk/2018/04/parallel-processing-pandas-dataframes/
    with multiprocessing.Pool(num_processes) as pool:
        results = list(
            tqdm(pool.imap(parallel_preprocess_func, df.iterrows()),
                 total=df.shape[0],
                 unit='files',
                 dynamic_ncols=True))

    time_taken = strftime('%H:%M:%S', gmtime(time() - time_start))
    print('Preprocessing took {} for {} files'.format(time_taken, df.shape[0]))
    # print('Preprocessing took {} for {} files ({})'.format(time_taken, df.shape[0], human_size(total_file_size)))
    # print('Dropped {} ({:.2%}) files ({} empty string, {} exceptions)'.format(dropped_files, dropped_files / total_file_num, dropped_empty, dropped_exception))

    return results


def get_email_body_from_file(email_path):
    return parse_from_file(email_path).body

    # a = ''

    # with open(email_path, 'r', encoding='utf-8', errors='ignore') as f:
    #     a = f.read()

    # # https://stackoverflow.com/a/36598450/3256255
    # # a = a.encode('ascii', errors='ignore')

    # # https://stackoverflow.com/a/32840516/3256255
    # if isinstance(a, str):
    #     b = email.message_from_string(a)
    # else:
    #     b = email.message_from_bytes(a)
    # body = ''

    # if b.is_multipart():
    #     for part in b.walk():
    #         ctype = part.get_content_type()
    #         cdispo = str(part.get('Content-Disposition'))

    #         # skip any text/plain (txt) attachments
    #         if ctype == 'text/plain' and 'attachment' not in cdispo:
    #             body = part.get_payload(decode=True)  # decode
    #             break
    # # not multipart - i.e. plain text, no attachments, keeping fingers crossed
    # else:
    #     body = b.get_payload(decode=True)

    # return body


def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove html tags and all line breaks, also extra whitespace
    text = strip_html(text).replace('\n', ' ').replace('\r', '').strip()

    # Expand contractions
    text = contractions.fix(text)

    # Remove all punctuation
    # https://stackoverflow.com/a/266162/3256255
    text = punc_regex.sub('', text)

    # Return
    return text


def strip_html(html):
    # lxml for speed, unsure for consequences
    return BeautifulSoup(html, 'lxml').text


def preprocess_text_tokenize(text):
    # Tokenize
    words = nltk.word_tokenize(text)

    for index, word in enumerate(words):
        # Remove stop words
        if word in cached_stopwords:
            words.pop(index)
        # Remove numbers
        elif word.isdigit():
            words.pop(index)
        # Lemmazation
        else:
            words[index] = lemmatizer.lemmatize(word)

    return words


def after_preprocess_clean(data):
    time_start = time()

    cleaned_data = [x for x in data if x is not None]

    time_taken = strftime('%H:%M:%S', gmtime(time() - time_start))
    print('After-preprocess cleaning took {} for {} rows ({} left)'.format(time_taken,
                                                                           len(data), len(cleaned_data)))
    return cleaned_data


# https://stackoverflow.com/a/43750422/3256255
def human_size(bytes, units=[' bytes', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB']):
    """ Returns a human readable string reprentation of bytes"""
    return str(bytes) + units[0] if bytes < 1024 else human_size(bytes >> 10, units[1:])


if __name__ == '__main__':
    preprocess(os.path.join(dataset_path, index_path))
