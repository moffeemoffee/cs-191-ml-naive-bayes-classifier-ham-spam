from mailparser import parse_from_file
from os import path
from threading import Thread
from tqdm import tqdm
import numpy as np
import pandas as pd
import re

exit_flag = 0

dataset_path = 'dataset/trec07p/'
index_path = 'full/index'
csv_path = 'processed.csv'

_cleanr = re.compile('<.*?>')


def preprocess(index):
    files = pd.read_csv(index, sep=' ', names=['is_spam', 'email_path'])
    files['is_spam'] = files['is_spam'].map({'spam': 1, 'ham': 0})
    files['text'] = ''

    results = _parallel_preprocess(files, 8)
    results.to_csv(path.join(dataset_path, csv_path))


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
        self.progress = 0
        self.total = len(files.index)
        self.pbar = pbar

    def run(self):
        # print('Starting {}'.format(self.name))
        self.read_emails()
        # print('Stopping {}'.format(self.name))

    def read_emails(self):
        for index, row in self.files.iterrows():
            self.progress += 1
            self.pbar.update(1)
            try:
                email_path = path.join(
                    dataset_path, index_path, '..', row['email_path'])
                email_path = path.abspath(email_path)
                self.files.at[index, 'text'] = get_email_body_from_path(
                    email_path)
            except Exception as e:
                print('Failed to read {}: {}'.format(row['email_path'], e))
                self.files.at[index, 'text'] = ''
                continue

        if exit_flag:
            self.name.exit()


def get_email_body_from_path(email_path):
    email_body = parse_from_file(email_path).body
    email_body = cleanhtml(email_body).replace('\n', ' ').strip()
    return email_body


# https://stackoverflow.com/questions/9662346/python-code-to-remove-html-tags-from-a-string
def cleanhtml(raw_html):
    return re.sub(_cleanr, '', raw_html)


if __name__ == '__main__':
    preprocess(path.join(dataset_path, index_path))
