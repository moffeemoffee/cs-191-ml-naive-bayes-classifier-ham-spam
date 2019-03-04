from naivebayes import NaiveBayes
from sklearn.model_selection import train_test_split
import os
import numpy as np
import pandas as pd

dataset_path = 'dataset/trec07p/'
csv_path = 'processed-2019-03-03-18-29-36.csv'

if __name__ == '__main__':
    # Read file
    df = pd.read_csv(os.path.join(dataset_path, csv_path),
                     header=0, index_col=0)

    # Split data
    targets = np.int64([0, 1])
    target_names = ['ham', 'spam']
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['is_spam'], test_size=0.2, random_state=191)

    # Build Classifier
    gvoc_model = NaiveBayes(X_train, y_train, targets, target_names)
    gvoc_model.train()

    gvoc_model.evaluate('General Vocabulary', X_test, y_test)

    rvoc_model = NaiveBayes(X_train, y_train, targets,
                            target_names, max_features=200)
    rvoc_model.train()

    rvoc_model.evaluate('Reduced Vocabulary', X_test, y_test)
