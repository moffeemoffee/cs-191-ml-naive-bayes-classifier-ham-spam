from bs4 import BeautifulSoup
from datetime import datetime
# from mailparser import parse_from_file
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from os import path, stat
from threading import Thread
from time import gmtime, sleep, strftime, time
# from time import time
from tqdm import tqdm
import contractions
import email
import logging
import lxml
import numpy as np
import nltk
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
wordnet_lemmatizer = WordNetLemmatizer()

total_file_size = 0
dropped_files = 0
dropped_empty = 0
dropped_exception = 0


def preprocess(index):
    time_start = time()
    files = pd.read_csv(index, sep=' ', names=['is_spam', 'email_path'])
    files['is_spam'] = files['is_spam'].map({'spam': 1, 'ham': 0})
    files['text'] = ''

    total_file_num = len(files.index)

    results = _parallel_preprocess(files, 8)
    results.to_csv(path.join(dataset_path, csv_path))

    time_end = time()
    time_taken = strftime('%H:%M:%S', gmtime(time_end - time_start))
    print('Preprocessing took {} for {} files ({})'.format(
        time_taken, total_file_num, human_size(total_file_size)))
    print('Dropped {} (0:.2%) files ({} empty string, {} exceptions)'.format(
        dropped_files, dropped_files / total_file_num, dropped_empty, dropped_exception))


def _parallel_preprocess(files, num_threads=2):
    threads = []
    thread_results = []
    total_files_num = len(files.index)

    with tqdm(total=total_files_num, unit='files', dynamic_ncols=True) as pbar:
        # idk if this should be here
        tqdm.write('Waiting for WordNet to load...')
        wordnet.ensure_loaded()

        # Split data, then multi-thread
        split_files = np.array_split(files, num_threads)
        for i in range(num_threads):
            new_thread = emailParseThread(
                i, 'emailParseThread{}'.format(i), split_files[i], pbar)
            threads.append(new_thread)
            new_thread.start()

        # Wait for threads to finish, then append their results
        for thread in threads:
            thread.join()
            thread_results.append(thread.files)

    return pd.concat(thread_results)


class emailParseThread(Thread):
    def __init__(self, thread_id, name, files, pbar):
        Thread.__init__(self)
        self.thread_id = thread_id
        self.name = name
        self.files = files
        self.pbar = pbar

    def run(self):
        # print('Starting {}'.format(self.name))
        self.read_emails()
        # print('Stopping {}'.format(self.name))

    def read_emails(self):
        global total_file_size
        global dropped_files
        global dropped_empty
        global dropped_exception
        for index, row in self.files.iterrows():
            # tqdm.write('Reading {}'.format(row['email_path']))
            try:
                email_path = path.join(
                    dataset_path, index_path, '..', row['email_path'])
                email_path = path.abspath(email_path)
                total_file_size += stat(email_path).st_size
                email_body = preprocess_text(
                    get_email_body_from_file(email_path))
                if not email_body:
                    # String is empty wtf
                    # tqdm.write('String at {} is empty, dropping'.format(row['email_path']))
                    dropped_files += 1
                    dropped_empty += 1
                    self.files.drop([index])
                else:
                    email_words = preprocess_text_tokenize(
                        email_body, wordnet_lemmatizer)
                    self.files.at[index, 'text'] = email_words
                self.pbar.update(1)
            except Exception as e:
                tqdm.write('Exception at {}'.format(row['email_path']))
                logging.exception('message')
                # self.files.at[index, 'text'] = ''
                dropped_files += 1
                dropped_exception += 1
                self.files.drop([index])
                self.pbar.update(1)
                tqdm.write('Deleted row {}'.format(row['email_path']))
                sleep(10)
                continue

        if exit_flag:
            self.name.exit()


def get_email_body_from_file(email_path):
    # return parse_from_file(email_path).body

    a = ''

    with open(email_path, 'r', encoding='utf-8', errors='ignore') as f:
        a = f.read()

    # https://stackoverflow.com/a/36598450/3256255
    # a = a.encode('ascii', errors='ignore')

    # https://stackoverflow.com/a/32840516/3256255
    if isinstance(a, str):
        b = email.message_from_string(a)
    else:
        b = email.message_from_bytes(a)
    body = ''

    if b.is_multipart():
        for part in b.walk():
            ctype = part.get_content_type()
            cdispo = str(part.get('Content-Disposition'))

            # skip any text/plain (txt) attachments
            if ctype == 'text/plain' and 'attachment' not in cdispo:
                body = part.get_payload(decode=True)  # decode
                break
    # not multipart - i.e. plain text, no attachments, keeping fingers crossed
    else:
        body = b.get_payload(decode=True)

    return body


def preprocess_text(text):
    # Remove html tags and all line breaks, also extra whitespace
    text = strip_html(text).replace('\n', ' ').replace('\r', '').strip()

    # Convert to lowercase
    text = text.lower()

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


def preprocess_text_tokenize(text, lemmatizer):
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


# https://stackoverflow.com/a/43750422/3256255
def human_size(bytes, units=[' bytes', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB']):
    """ Returns a human readable string reprentation of bytes"""
    return str(bytes) + units[0] if bytes < 1024 else human_size(bytes >> 10, units[1:])


if __name__ == '__main__':
    preprocess(path.join(dataset_path, index_path))
