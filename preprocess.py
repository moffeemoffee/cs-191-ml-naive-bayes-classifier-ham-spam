import pandas as pd
import numpy as np
import mailparser
import os
import threading

exit_flag = 0

dataset_path = 'dataset/trec07p/'
index_path = 'full/index'
csv_path = 'processed.csv'


def preprocess(index):
    files = pd.read_csv(index, sep=' ', names=['is_spam', 'email_path'])
    files['is_spam'] = files['is_spam'].map({'spam': 1, 'ham': 0})
    files['text'] = ''

    results = _parallel_preprocess(files, 8)
    results.to_csv(os.path.join(dataset_path, csv_path))


def _parallel_preprocess(files, num_threads=2):
    threads = []
    thread_results = []

    split_files = np.array_split(files, num_threads)
    for i in range(num_threads):
        new_thread = emailParseThread(
            i, 'emailParseThread{}'.format(i), split_files[i])
        threads.append(new_thread)
        new_thread.start()

    for thread in threads:
        thread.join()
        thread_results.append(thread.files)

    return pd.concat(thread_results)


class emailParseThread(threading.Thread):
    def __init__(self, thread_id, name, files):
        threading.Thread.__init__(self)
        self.thread_id = thread_id
        self.name = name
        self.files = files

    def run(self):
        print('Starting {}'.format(self.name))
        self.read_emails()
        print('Stopping {}'.format(self.name))

    def read_emails(self):
        for index, row in self.files.iterrows():
            print(row['email_path'])
            try:
                self.files.at[index, 'text'] = extract_email_from_file(
                    os.path.join(base_dir, row['email_path']))
            except:
                self.files.at[index, 'text'] = ''
                continue

        if exit_flag:
            self.name.exit()


def extract_email_from_file(email_file):
    mail = mailparser.parse_from_file(email_file)
    return cleanhtml(mail.body).replace('\n', ' ').strip()


# https://stackoverflow.com/questions/9662346/python-code-to-remove-html-tags-from-a-string
def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext


if __name__ == '__main__':
    preprocess(os.path.join(dataset_path, index_path))
