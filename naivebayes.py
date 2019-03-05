from preprocess import preprocess_text
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from time import gmtime, strftime, time
from tqdm import tqdm
import math
import numpy as np


def get_feature_prob(feat_count, class_total_counts):
    return np.log(feat_count / float(class_total_counts))


def get_feature_prob_laplace(feat_count, class_total_counts, feat_number):
    return np.log((feat_count + 1) / float(class_total_counts + feat_number))


class NaiveBayes:

    def __init__(self, name, X_train, y_train, targets, target_names, max_features=None):
        self.name = name
        self.X_train = X_train
        self.y_train = y_train
        self.targets = targets
        self.target_names = target_names
        self.max_target_name_len = len(max(target_names, key=len))
        self.max_features = max_features

        # Data to be assigned later
        self.target_data = {}
        self.total_counts = 0
        self.feat_number = 0

    def train(self, num_processes=None):
        time_start = time()

        print('Training {}...'.format(self.name))

        self.total_counts = 0
        # TODO: Use get_feature_names() instead on each CountVectorizer and append to
        # a set, and get the count
        self.feat_number = CountVectorizer(max_features=self.max_features).fit_transform(
            self.X_train).sum(axis=0).shape[1]

        for t in tqdm(self.targets, unit='class'):
            X_train_target = [x for x, y in zip(
                self.X_train, self.y_train) if y == t]

            t_data = {}
            t_data['vec'] = CountVectorizer(max_features=self.max_features)
            t_data['bow'] = t_data['vec'].fit_transform(
                X_train_target).sum(axis=0)
            t_data['bow_count'] = np.sum(t_data['bow'])
            t_data['feat_count'] = t_data['bow'].shape[1]
            t_data['doc_count'] = len(X_train_target)
            t_data['prior'] = np.log(
                t_data['doc_count'] / float(self.X_train.shape[0]))
            # t_data['data'] = X_train_target

            self.total_counts += t_data['bow_count']
            self.target_data[t] = t_data

        for t, t_name in zip(self.targets, self.target_names):
            t_data = self.target_data[t]
            t_data['likelihood'] = {}
            t_data['lap_likelihood'] = {}
            tqdm_desc = '{:{width}} Likelihood'.format(
                t_name, width=self.max_target_name_len)
            tqdm_list = tqdm(t_data['vec'].vocabulary_.items(),
                             desc=tqdm_desc,
                             unit='word')
            for word, idx in tqdm_list:
                prob = get_feature_prob(t_data['bow'][0, idx],
                                        t_data['bow_count'])
                lap_prob = get_feature_prob_laplace(t_data['bow'][0, idx],
                                                    t_data['bow_count'],
                                                    self.feat_number)
                t_data['likelihood'][word] = prob
                t_data['lap_likelihood'][word] = lap_prob

        print('Training complete with {} features ({} total counts)'.format(
            self.feat_number, self.total_counts))

        time_taken = strftime('%H:%M:%S', gmtime(time() - time_start))
        print('Training took {}.\n'.format(time_taken))

    # Worse version
    def predict_old(self, X_test):
        predicted = []
        print('Testing..')

        for test_case in tqdm(X_test):
            target_sums = [self.target_data[t]['prior']
                           for t in self.targets]
            test_words = preprocess_text(test_case)

            for t_idx, t in enumerate(self.targets):
                t_data = self.target_data[t]
                for word in test_words:
                    if word in t_data['likelihood']:
                        target_sums[t_idx] += t_data['likelihood'][word]

            # Get biggest result
            predicted.append(self.targets[np.argmax(target_sums)])

        return predicted

    # Uses smoothing
    def predict(self, X_test):
        print('Testing {}...'.format(self.name))
        predicted = []
        lap_predicted = []

        smooth_probs = [math.log(1/(t_data['doc_count'] + len(self.X_train)))
                        for t, t_data in self.target_data.items()]

        # TODO: multiprocessing
        for test_case in tqdm(X_test, unit='test'):
            target_sums = [self.target_data[t]['prior']
                           for t in self.targets]
            lap_target_sums = [self.target_data[t]['prior']
                               for t in self.targets]
            test_words = preprocess_text(test_case)

            for t_idx, t in enumerate(self.target_data):
                t_data = self.target_data[t]
                for word in test_words:
                    if word in t_data['likelihood']:
                        target_sums[t_idx] += t_data['likelihood'][word]
                        lap_target_sums[t_idx] += t_data['lap_likelihood'][word]
                    else:
                        for t2_idx, t2 in enumerate(self.targets):
                            t2_data = self.target_data[t2]
                            if t2 != t and word in t2_data['likelihood']:
                                target_sums[t_idx] += smooth_probs[t_idx]
                                lap_target_sums[t_idx] += smooth_probs[t_idx]

            # Get biggest result
            predicted.append(self.targets[np.argmax(target_sums)])
            lap_predicted.append(self.targets[np.argmax(lap_target_sums)])

        return predicted, lap_predicted

    def _evaluate(self, label, predicted, y_test):
        print(label)
        print('Accuracy: {:.05%}'.format(np.mean(predicted == y_test)))
        print(metrics.classification_report(y_test,
                                            predicted,
                                            target_names=self.target_names))

    def evaluate(self, X_test, y_test, show_top_features=False):
        predicted, lap_predicted = self.predict(X_test)

        self._evaluate('\n{} Without Laplace Smoothing'.format(self.name),
                       predicted, y_test)
        self._evaluate('{} With Laplace Smoothing'.format(self.name),
                       lap_predicted, y_test)

        if show_top_features is True:
            show_top_features = 10
        if isinstance(show_top_features, int) and show_top_features > 0:
            fancy_table = {}
            max_lengths = {}
            for t_name, t_data_idx in zip(self.target_names, self.target_data):
                max_len = {'word': 0, 'count': 0}

                t_data = self.target_data[t_data_idx]

                t_words = [(word, t_data['bow'][0, idx])
                           for word, idx in t_data['vec'].vocabulary_.items()]

                # Descending sort, obtain only the top <show_top_features>
                t_words = sorted(t_words,
                                 key=lambda x: x[1],
                                 reverse=True)[:show_top_features]

                for t_tup in t_words:
                    word = t_tup[0]
                    count = t_tup[1]
                    word_len = len(str(word))
                    count_len = len(str(count))
                    if word_len > max_len['word']:
                        max_len['word'] = word_len
                    if count_len > max_len['count']:
                        max_len['count'] = count_len

                fancy_table[t_name] = t_words
                max_lengths[t_name] = max_len

            # Print headers
            print('Top {} words:'.format(show_top_features))

            for t_name in fancy_table:
                w_width = max_lengths[t_name]['word']
                c_width = max_lengths[t_name]['count']
                print('{} {:{w_width}}'.format(
                    ' ' * c_width,
                    t_name,
                    w_width=w_width + 1), end='| ')
            print('')

            # Print contents
            for i in range(show_top_features):
                for t_name, t_words in fancy_table.items():
                    word = t_words[i][0]
                    count = t_words[i][1]
                    print('{c:>{c_width}} {w:{w_width}} '.format(
                        w=word, c=count,
                        w_width=max_lengths[t_name]['word'],
                        c_width=max_lengths[t_name]['count']), end='| ')
                print('')
            print('')
