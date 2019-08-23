import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

# Adjust this to select how many words are used for classification. You'll notice as your increase from 50 - 1000,
# accuracy goes up, however, so does the runtime.
no_of_retained_words = 200


# Opens an email from it's file name, returns the 3rd line in the email (where all the content is).
def get_line_from_file(filename, n):
    with open(filename) as file:
        for i, line in enumerate(file):
            if i == n:
                return line


# Creates a dictionary from the emails.
def make_dictionary(train_files):

    emails = train_files
    all_words = []

    # Go through emails, add all words to dictionary.
    for mail in emails:
        line = get_line_from_file(mail, 2)
        words = line.split()
        all_words += words

    # Elements stored as dictionary keys and counts stores as values.
    dictionary = Counter(all_words)

    # Removes duplicates.
    list_to_remove = list(dictionary)

    # Removes punctuation and single character words.
    for item in list_to_remove:
        if item.isalpha() is False:
            del dictionary[item]
        elif len(item) == 1:
            del dictionary[item]

    # Returns a list of the most common words and their counts.
    dictionary = dictionary.most_common(no_of_retained_words)

    return dictionary


# This marks emails as "spam" (1) or "ham/legit" (0).
def build_labels(files):
    emails = files
    emails.sort()

    labels_matrix = np.zeros(len(emails))

    for index, email in enumerate(emails):
        labels_matrix[index] = 1 if 'spms' in email else 0

    return labels_matrix


# This is where the word occurrences are counted.
def extract_features(files, dictionary):

    # Array of file names
    emails = files
    # Create a blank (array files x dictionary)
    features_matrix = np.zeros((len(emails), len(dictionary)))

    # This places the "word" as the key, making it quickly searchable, and places the "index", as the value, useful
    # later. This was necessary for performance.
    dictionary = dict([(d[0], i) for i, d in enumerate(dictionary)])

    # Go through each email, line by line, check if the words are in the dictionary, build feature matrix.
    for doc_id, fil in enumerate(emails):
        line = get_line_from_file(fil, 2)
        words = line.split()
        for word in words:
            if word in dictionary:
                word_id = dictionary[word]
                # We're essentially seeing what words are common in spam emails, and non-spam emails. We're placing
                # these observations in features_matrix.
                features_matrix[doc_id, word_id] = words.count(word)
    return features_matrix


# This method will open up a parent folder and extract the emails from each part within.
def run_bayesian(parent_folder):
    print("Running", parent_folder)

    folder_names = []

    # Look through parent folder.
    for root, dirs, files in os.walk(parent_folder, topdown=False):
        for name in dirs:
            # Grabs parts 1 - 10 (which will be our k-fold chunks)
            folder_names.append(os.path.join(root, name))

    acc_score = []
    spam_prec_score = []
    spam_recall_score = []

    # With parts 1-10, cycle through 10 times, each iteration choosing a different test folder.
    for i in range(len(folder_names)):

        test_dir = folder_names[i]
        train_files = []
        test_files = []

        for j in range(len(folder_names)):
            # Grab all files from train folders and put into one array.
            if folder_names[j] != test_dir:
                for root, dirs, files in os.walk(folder_names[j], topdown=False):
                    for name in files:
                        train_files.append(os.path.join(root, name))
            # Put all test files into one array.
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
        acc_score.append(accuracy_score(test_labels, result1))
        spam_prec_score.append(precision_score(test_labels, result1))
        spam_recall_score.append(recall_score(test_labels, result1, pos_label=1, average='binary'))

    avg_acc = np.mean(acc_score)
    # avg_spam_prec = np.mean(spam_prec_score)
    # avg_spam_rec = np.mean(spam_recall_score)

    # print("10-fold Validation Complete...")
    # print("Average accuracy: ")
    # print("%.2f" % avg_acc, "%")
    # print("Average spam precision: ")
    # print("%.2f" % avg_spam_prec, "%")
    # print("Average spam recall: ")
    # print("%.2f" % avg_spam_rec, "%")

    return spam_recall_score, spam_prec_score, avg_acc


if __name__ == "__main__":
    spam_rec, spam_prec, acc = run_bayesian('bare')
    print("Average accuracy for bare: ", acc)
    plt.scatter(spam_rec, spam_prec, marker='o', label="no stop list, no lemmatizer")
    spam_rec, spam_prec, acc = run_bayesian('lemm')
    plt.scatter(spam_rec, spam_prec, marker='x', label="no stop list, lemmatizer")
    print("Average accuracy for lemm: ", acc)
    spam_rec, spam_prec, acc = run_bayesian('stop')
    plt.scatter(spam_rec, spam_prec, marker='+', label="stop list, no lemmatizer")
    print("Average accuracy for stop: ", acc)
    spam_rec, spam_prec, acc = run_bayesian('lemm_stop')
    plt.scatter(spam_rec, spam_prec, marker='^', label="stop list, lemmatizer")
    print("Average accuracy for lemm_stop: ", acc)

    plt.xlim([0.6, 1.0])
    plt.ylim([0.6, 1.0])
    plt.xlabel("spam recall")
    plt.ylabel("spam precision")
    plt.legend(loc='lower left', fontsize="small")
    plt.show()

# Spam recall: spam as spam / (spam as spam + spam as legit) spam message pass the filter, higher = less false negatives
# Spam precision: spam as spam / (spam as spam + legit as spam) we lose legit message, higher = less false positives




