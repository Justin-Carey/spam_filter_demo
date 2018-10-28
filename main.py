import os
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


def make_dictionary(train_files):
    emails = train_files
    all_words = []
    for mail in emails:
        with open(mail) as m:
            for i, line in enumerate(m):
                if i == 2:
                    words = line.split()
                    all_words += words

    dictionary = Counter(all_words)
    list_to_remove = list(dictionary)
    for item in list_to_remove:
        if item.isalpha() == False:
            del dictionary[item]
        elif len(item) == 1:
            del dictionary[item]
    dictionary = dictionary.most_common(3000)
    return dictionary


def extract_features(file_dir, dictionary):
    files = file_dir
    features_matrix = np.zeros((len(files), 3000))
    doc_id = 0
    for fil in files:
        with open(fil) as fi:
            for i, line in enumerate(fi):
                if i == 2:
                    words = line.split()
                    for word in words:
                        word_id = 0
                        for j, d in enumerate(dictionary):
                            if d[0] == word:
                                word_id = j
                                features_matrix[doc_id, word_id] = words.count(word)
            doc_id = doc_id + 1
    return features_matrix


def run_bayesian(parent_folder):

    folder_names = []

    for root, dirs, files in os.walk(parent_folder, topdown=False):
        for name in dirs:
            folder_names.append(os.path.join(root, name))

    for i in range(len(folder_names)):

        test_dir = folder_names[i]
        train_files = []
        test_files = []
        spam_count = 0
        ham_count = 0
        spam_array = []
        ham_array = []
        test_spam_count = 0
        test_ham_count = 0
        test_spam_array = []
        test_ham_array = []

        for j in range(len(folder_names)):
            if folder_names[j] != test_dir:
                for root, dirs, files in os.walk(folder_names[i], topdown=False):
                    for name in files:
                        if 'spmsga' in name:
                            spam_count = spam_count + 1
                            spam_array.append(os.path.join(root, name))
                        else:
                            ham_count = ham_count + 1
                            ham_array.append(os.path.join(root, name))
            else:
                for root, dirs, files in os.walk(folder_names[i], topdown=False):
                    for name in files:
                        if 'spmsga' in name:
                            test_spam_count = test_spam_count + 1
                            test_spam_array.append(os.path.join(root, name))
                        else:
                            test_ham_count = ham_count + 1
                            test_ham_array.append(os.path.join(root, name))

        train_files = np.concatenate((spam_array, ham_array))
        test_files = np.concatenate((test_spam_array, test_ham_array))
        total = ham_count + spam_count
        test_total = test_ham_count + test_spam_count

        dictionary = make_dictionary(train_files)

        train_labels = np.zeros(total)
        train_labels[spam_count:total-1] = 1
        train_matrix = extract_features(train_files, dictionary)

        model1 = MultinomialNB()
        model1.fit(train_matrix, train_labels)

        test_labels = np.zeros(test_total)
        test_labels[test_spam_count:test_total] = 1
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




