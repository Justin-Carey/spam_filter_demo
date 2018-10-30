import os
import numpy as np
from collections import Counter
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import re


# Creates a dictionary from the emails.
def make_dictionary(train_files):
    # print("Entering make_dictionary..")
    # Put email names into emails
    emails = train_files
    all_words = []
    # Go through emails, add all words to dictionary.
    for mail in emails:
        with open(mail) as m:
            for i, line in enumerate(m):
                if i == 2:
                    words = line.split()
                    all_words += words

    # Elements stores as dictionary keys and counts stores as values.
    dictionary = Counter(all_words)
    # Removes duplicates.
    list_to_remove = list(dictionary)

    for item in list_to_remove:
        # Removes puncuation etc.
        if item.isalpha() == False:
            del dictionary[item]
        elif len(item) == 1:
            del dictionary[item]
    # Returns a list of the N most common words and their counts.
    dictionary = dictionary.most_common(3000)
    return dictionary


def get_line_from_file(filename, n):
    with open(filename) as file:
        for i, line in enumerate(file):
            if i == n:
                return line


# This is where the word occurrences are counted.
def extract_features(file_dir, dictionary):

    # print("Entering extract_features..")

    # Array of filenames
    files = file_dir
    # Create a blank array files x dictionary
    features_matrix = np.zeros((len(files), len(dictionary)))
    doc_id = 0

    dictionary = [(k, v) for k, v in enumerate(dictionary)]

    for fil in files:
        line = get_line_from_file(fil, 2)
        words = line.split()
        for word in words:
            word_id = 0
            if word in dictionary:
                word_id = dictionary[word]
                features_matrix[doc_id, word_id] = words.count(word)
    doc_id = doc_id + 1

    return features_matrix


def build_labels(files):
    # print("Entering build_labels..")
    emails = files
    emails.sort()

    labels_matrix = np.zeros(len(emails))

    for index, email in enumerate(emails):
        labels_matrix[index] = 1 if re.search('spms*', email) else 0

    return labels_matrix


# This method will open up a parent folder and extract the emails from each part within.
def run_bayesian(parent_folder):

    folder_names = []

    for root, dirs, files in os.walk(parent_folder, topdown=False):
        for name in dirs:
            # Grabs parts 1 - 10 (which will be our k fold chunks)
            folder_names.append(os.path.join(root, name))

    # With parts 1-10, cycle through 10 times, each iteration choosing a different test folder.
    for i in range(len(folder_names)):

        test_dir = folder_names[i]
        train_files = []
        test_files = []

        for j in range(len(folder_names)):
            if folder_names[j] != test_dir:
                for root, dirs, files in os.walk(folder_names[j], topdown=False):
                    for name in files:
                        train_files.append(os.path.join(root, name))
            else:
                for root, dirs, files in os.walk(folder_names[j], topdown=False):
                    for name in files:
                        test_files.append(os.path.join(root, name))

        dictionary = make_dictionary(train_files)

        train_labels = build_labels(train_files)
        train_matrix = extract_features(train_files, dictionary)

        model1 = MultinomialNB()
        model1.fit(train_matrix, train_labels)

        test_labels = build_labels(test_files)
        test_matrix = extract_features(test_files, dictionary)

        result1 = model1.predict(test_matrix)
        cm = confusion_matrix(test_labels, result1)
        acc_score = accuracy_score(test_labels, result1)

        acc_score = acc_score * 100

        print("%.2f" % acc_score, "% accuracy")
        print(cm)


if __name__ == "__main__":
    run_bayesian('bare')
    # run_bayesian('lemm')
    # run_bayesian('stop')
    # run_bayesian('lemm_stop')




