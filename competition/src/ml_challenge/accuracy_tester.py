#! /usr/bin/env python3
import argparse
import os
import csv

from classifier import Classifier


def load_labels(metadata_path):
    labels = {}
    with open(metadata_path, 'r') as metadata_file:
        reader = csv.reader(metadata_file)
        for line in reader:
            labels[int(line[0])] = line[1]
    return labels


def metadata_path_from_dataset_path(dataset_path):
    if not os.path.isfile(dataset_path):
        raise ValueError('Invalid dataset file %s' % dataset_path)
    dataset_dir = os.path.split(dataset_path)[0]
    return os.path.join(dataset_dir, 'meta.csv')


def test_classifier_accuracy(classifier, dataset_path):
    n_data, n_positives = 0, 0
    with open(dataset_path, 'r') as dataset:
        reader = csv.reader(dataset)
        for (communication, label) in reader:
            n_data += 1
            try:
                predicted_label = classifier.classify(communication)
                if predicted_label == int(label):
                    n_positives += 1
            except:
                continue

    return n_positives / n_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', '-d', help='Path of the dataset to test', required=True)
    parser.add_argument('--metadata_path', '-m', help='Path of the metadata of the dataset. Assumed to be next to the dataset by default')
    args = parser.parse_args()

    metadata_path = args.metadata_path
    if metadata_path is None:
        metadata_path = metadata_path_from_dataset_path(args.dataset_path)

    labels = load_labels(metadata_path)
    classifier = Classifier(labels)
    accuracy = test_classifier_accuracy(classifier, args.dataset_path)
    print(accuracy)

