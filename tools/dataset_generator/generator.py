#! /usr/bin/env python3
import argparse
import csv
import os
import glob
import random
import math

help_string = """Generate a dataset in csv format from twitter raw csv dumps. Takes as an
argument the directory in which are located the raw csv dumps. The first level
of this directory is expected to only contains sub-directories (no csv files).
Each of those subdirectories is a class of the dataset, the name of the class
being the name of the directory. """


def raw_csv_file_paths_per_class(input_directory):
    dataset_files = {}
    classes_dirs  = os.listdir(input_directory)
    for c in classes_dirs:
        full_path = os.path.join(input_directory, c)
        if not os.path.isdir(full_path):
            raise ValueError('Invalid dataset directory format')
        files = [f for f in glob.glob(os.path.join(full_path, "*.csv"))]
        dataset_files[c] = files
    return dataset_files


def files_to_dataset(files_dirctionary):
    dataset = []
    classes = []
    for class_id, (clazz, files) in enumerate(files_dirctionary.items()):
        classes.append(clazz)
        for f in files:
            dataset += [(tweet, class_id) for tweet in load_raw_csv_file(f)]
    random.shuffle(dataset)
    return dataset, classes


def load_raw_csv_file(path):
    with open(path, 'r') as raw_csv:
        csv_reader = csv.reader(raw_csv)
        return [tweet[-1] for tweet in csv_reader][1:]


def write_metadata(classes, output_directory):
    with open(os.path.join(output_directory, 'meta.csv'), 'w') as csv_file:
        writer = csv.writer(csv_file)
        for i, c in enumerate(classes):
            writer.writerow([i, c])


def write_dataset_split(split_name, data, output_directory):
    with open(os.path.join(output_directory, split_name + '.csv'), 'w') as csv_file:
        csv.writer(csv_file).writerows(data)


def save_dataset(dataset, classes, train_split, output_directory):
    train_split = math.floor(args.train_split * len(dataset))
    train_dataset = dataset[:train_split]
    test_dataset = dataset[train_split:]
    write_dataset_split('train', train_dataset, output_directory)
    write_dataset_split('test', test_dataset, output_directory)
    write_metadata(classes, output_directory)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=help_string)
    parser.add_argument('--input_directory', '-i', type=str, required=True)
    parser.add_argument('--output_directory', '-o', type=str, required=True)
    parser.add_argument('--train_split', '-t', type=float, help='float between 0 and 1 indicating the percentage of data to put in the train set', required=True)
    args = parser.parse_args()

    files = raw_csv_file_paths_per_class(args.input_directory)
    dataset, classes = files_to_dataset(files)
    save_dataset(dataset, classes, args.train_split, args.output_directory)



