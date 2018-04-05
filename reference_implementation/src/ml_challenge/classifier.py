import csv
import sys
import re
import pickle
import numpy as np
import math


class Classifier:
    """Basic Naïve Bayes classifier based on instructions provided during the
       competition."""

    def __init__(self, label_dic):
        """
        Args:
        - label_dic :  dictionary to convert between int labels and string labels.
        """
        self.label_dic = label_dic
        self.n_classes = len(label_dic)
        with open('./naive_bayes.pkl', 'rb') as f:
            self.priors, self.probs = pickle.load(f)
        print(self.priors)

    def _words_log_probabilities(self, communication, prob_map):
        for w in words_of_communication(communication):
            if w in prob_map:
                prob = prob_map[w]
                yield math.log(prob)

    def classify(self, communication):
        best_class, best_likelihood = None, -math.inf
        for clazz, probs in self.probs.items():
            likelihood = math.log(self.priors[clazz])
            for log_prob in self._words_log_probabilities(
                    communication, probs):
                likelihood += log_prob
            if likelihood > best_likelihood:
                best_class = clazz
                best_likelihood = likelihood
        return best_class


def load_dataset(dataset_path):
    with open(dataset_path, 'r') as dataset_file:
        reader = csv.reader(dataset_file)
        dataset = [(com, int(label)) for com, label in reader]
        return dataset


def words_of_communication(communication):
    words = []
    for w in communication.split():
        w = w.lower()
        match = re.search('[a-z]{3,}', w)
        if match:
            words.append(match[0])
    return words


def counts_to_prob_arrays(counts, vocabulary):
    n_words = np.sum(np.array(list(counts.values()), dtype=int))
    probs = {}
    for word in vocabulary:
        count = 0 if word not in counts else counts[word]
        probs[word] = (count + 1) / (n_words + len(vocabulary))
    return probs


def train(dataset_path):
    dataset = load_dataset(dataset_path)
    word_counts = {}
    vocabulary = set()
    doc_counts = {clazz: 0 for _, clazz in dataset}
    n_doc = 0
    for data, clazz in dataset:
        n_doc += 1
        doc_counts[clazz] += 1

        if clazz not in word_counts:
            word_counts[clazz] = {}
        for word in words_of_communication(data):
            vocabulary.add(word)
            if word not in word_counts[clazz]:
                word_counts[clazz][word] = 1
            else:
                word_counts[clazz][word] += 1

    priors = {}
    for clazz, count in doc_counts.items():
        priors[clazz] = count / n_doc

    probs = {}
    for clazz, counts in word_counts.items():
        probs[clazz] = counts_to_prob_arrays(counts, vocabulary)

    with open('./naive_bayes.pkl', 'wb') as f:
        pickle.dump([priors, probs], f)

    print('Done training on %d documents' % n_doc)


if __name__ == '__main__':
    dataset_path = sys.argv[1]
    train(dataset_path)
