from naivebayes import NaiveBayes
from sklearn.model_selection import train_test_split
import os
import numpy as np
import pandas as pd

csv_path = 'processed.csv'

if __name__ == '__main__':
    # Read file
    df = pd.read_csv(os.path.join(csv_path), header=0, index_col=0)

    # Split data
    targets = np.int64([0, 1])
    target_names = ['ham', 'spam']
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['is_spam'], test_size=0.2, random_state=191)

    # Build Classifier
    gvoc_model = NaiveBayes('General Vocabulary', X_train,
                            y_train, targets, target_names)
    gvoc_model.train()

    gvoc_model.evaluate(X_test, y_test, show_top_features=10)

    rvoc_model = NaiveBayes('Reduced Vocabulary', X_train, y_train, targets,
                            target_names, max_features=200)
    rvoc_model.train()

    rvoc_model.evaluate(X_test, y_test, show_top_features=10)
