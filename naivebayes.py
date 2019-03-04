from preprocess import preprocess_text
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from time import gmtime, strftime, time
from tqdm import tqdm
import math
import multiprocessing as mp
import numpy as np


def get_feature_prob(feat_count, class_total_counts):
    return np.log(feat_count / float(class_total_counts))


def get_feature_prob_laplace(feat_count, class_total_counts, feat_number):
    return np.log((feat_count + 1) / float(class_total_counts + feat_number))


class NaiveBayes:

    def __init__(self, X_train, y_train, targets, target_names, max_features=None):
        self.X_train = X_train
        self.y_train = y_train
        self.targets = targets
        self.target_names = target_names
        self.max_features = max_features

        # Data to be assigned later
        self.target_data = {}
        self.total_counts = 0
        self.feat_number = 0

    def train(self, num_processes=None):
        time_start = time()

        print('Training...')

        self.total_counts = 0
        self.feat_number = CountVectorizer(max_features=self.max_features).fit_transform(
            self.X_train).sum(axis=0).shape[1]

        for t in tqdm(self.targets):
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
            for word, idx in t_data['vec'].vocabulary_.items():
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
        print('Testing..')
        predicted = []
        lap_predicted = []

        smooth_probs = [math.log(1/(t_data['doc_count'] + len(self.X_train)))
                        for t, t_data in self.target_data.items()]

        for test_case in tqdm(X_test):
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

    def evaluate(self, model_name, X_test, y_test):
        predicted, lap_predicted = self.predict(X_test)

        print('{} Without Laplace Smoothing'.format(model_name))
        print('Accuracy: {:.05%}'.format(np.mean(predicted == y_test)))
        print(metrics.classification_report(
            y_test, predicted, target_names=self.target_names))

        print('{} With Laplace Smoothing'.format(model_name))
        print('Accuracy: {:.05%}'.format(np.mean(lap_predicted == y_test)))
        print(metrics.classification_report(
            y_test, lap_predicted, target_names=self.target_names))
