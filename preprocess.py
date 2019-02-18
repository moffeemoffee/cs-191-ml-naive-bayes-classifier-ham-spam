from bs4 import BeautifulSoup
from datetime import datetime
from mailparser import parse_from_file
from os import path, stat
from threading import Thread
from time import sleep, time
from tqdm import tqdm
import lxml
import numpy as np
import pandas as pd
import re
import string

exit_flag = 0

dataset_path = 'dataset/trec07p/'
index_path = 'full/index'
csv_path = 'processed-{}.csv'.format(
    datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))

_cleanr = re.compile('<.*?>')

total_file_size = 0


def preprocess(index):
    time_start = time()
    files = pd.read_csv(index, sep=' ', names=['is_spam', 'email_path'])
    files['is_spam'] = files['is_spam'].map({'spam': 1, 'ham': 0})
    files['text'] = ''

    results = _parallel_preprocess(files, 8)
    results.to_csv(path.join(dataset_path, csv_path))

    print('Preprocessing took {} seconds for {} files ({} bytes)'.format(
        time() - time_start, len(files.index), human_size(total_file_size)))


def _parallel_preprocess(files, num_threads=2):
    threads = []
    thread_results = []
    total_files_num = len(files.index)

    with tqdm(total=total_files_num, unit='files') as pbar:
        split_files = np.array_split(files, num_threads)
        for i in range(num_threads):
            new_thread = emailParseThread(
                i, 'emailParseThread{}'.format(i), split_files[i], pbar)
            threads.append(new_thread)
            new_thread.start()

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
        for index, row in self.files.iterrows():
            try:
                email_path = path.join(
                    dataset_path, index_path, '..', row['email_path'])
                email_path = path.abspath(email_path)
                total_file_size += stat(email_path).st_size
                email_body = parse_from_file(email_path).body
                email_body = clean_html(email_body).replace('\n', ' ').strip()
                self.files.at[index, 'text'] = email_body
                self.pbar.update(1)
            except Exception as e:
                print('Exception at {}: {}'.format(row['email_path'], e))
                sleep(1)
                self.files.at[index, 'text'] = ''
                self.pbar.update(1)
                continue

        if exit_flag:
            self.name.exit()


# https://stackoverflow.com/questions/9662346/python-code-to-remove-html-tags-from-a-string
# Needs 4.5gb download lol: https://rushter.com/blog/python-fast-html-parser/
def clean_html(html):
    # return re.sub(_cleanr, '', html)
    return BeautifulSoup(html, 'lxml').text


def preprocess_text(text):
    # Remove punctuation
    # https://stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string-in-python
    text = text.translate(None, string.punctuation)
    return text


# https://stackoverflow.com/a/43750422/3256255
def human_size(bytes, units=[' bytes', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB']):
    """ Returns a human readable string reprentation of bytes"""
    return str(bytes) + units[0] if bytes < 1024 else human_size(bytes >> 10, units[1:])


if __name__ == '__main__':
    preprocess(path.join(dataset_path, index_path))
