import re
import numpy as np
from pathlib import Path


SMALL_DATA_PATH = Path(__file__).parent / "data" / "SmallDataset"

def load_small(filename):
    """
    Loads the small dataset used for the Naive Bayes Classifier. 

    params:
    - filename: file name for the small dataset.

    return: List of tuples of the label and the text [(label, text), ...]
    """

    # storing all results
    results = []

    # opening a file using with so it closes automatically. 'r' for read. 'utf-8' for specific encoding.
    with open(SMALL_DATA_PATH / filename, 'r', encoding='utf-8') as file:
        # reads line by line
        for line in file:
            # checks if the line is present or if it is an empty line
            if line:
                line = line.replace("\n", "").strip() # removes binding whitespace & newlines
                label, text = line[:3], line[4:] # splits the line into the label and the text
                results.append((label, text)) 
            else:
                continue # if the line is empty it continues the loop

    return results

def preprocess(text):
    text = re.sub(r"[?!,.]", '', text)
    text = text.lower()
    return text.split()

class NaiveBayesClassifier:
    def __init__(self):
        # stores the number of reviews per class
        self.documents = {}
        # how often each word appears in each class
        self.word_count = {}
        # how many words each class has in total, repeats are included
        self.total_word_count = {}
        # every unique wor seen in training
        self.vocabulary = set()

    def train(self, docs):
        for doc in docs:
            label, text = doc[0], doc[1]

            # preprocess the text
            text = preprocess(text)

            # adding class counts to documents
            if label in self.documents:
                self.documents[label] += 1
            else:
                self.documents[label] = 1

            # adding the different classes with their dictionaries into word_count
            if label not in self.word_count:
                self.word_count[label] = {}

            for word in text:
                if word in self.word_count[label]:
                    self.word_count[label][word] += 1
                else:
                    self.word_count[label][word] = 1

                if label in self.total_word_count:
                    self.total_word_count[label] += 1
                else:
                    self.total_word_count[label] = 1

                self.vocabulary.add(word)

    def prior(self, c):
        class_docs = self.documents[c]
        total_docs = sum(self.documents.values())
        return class_docs / total_docs

    def likelihood(self, word, c):
        num_in_class = self.word_count[c].get(word, 0) + 1 # La Place smooting
        total_in_class = self.total_word_count[c] + len(self.vocabulary)
        return num_in_class / total_in_class

    def score(self, words, c):
        result = 0

        p = np.log(self.prior(c))

        for word in words:
            if word in self.vocabulary:
                result += np.log(self.likelihood(word, c))

        return p + result

    def classify(self, text):
        classes = self.documents.keys()

        text = preprocess(text)

        ranking = {}

        for c in classes:
            s = self.score(text, c)
            ranking[c] = s

        return max(ranking, key=ranking.get)


def evaluate(model:NaiveBayesClassifier, test_docs): 

    pairs = []

    for label, text in test_docs:
        prediction = model.classify(text)
        pairs.append((label, prediction))

        marker = "" if prediction == label else "   <-- WRONG"
        print(f"{text} : {prediction} (true: {label}){marker}")

    return pairs

def metrics(pairs):
    # accuracy
    correct = 0
    for true, pred in pairs:
        if true == pred:
            correct += 1
    print("accuracy:", correct / len(pairs))

    # precision, recall, f1 for each class
    for c in ["POS", "NEU", "NEG"]:
        tp = 0
        fp = 0
        fn = 0

        for true, pred in pairs:
            if true == c and pred == c:
                tp += 1
            elif pred == c:          # predicted c, but it wasn't
                fp += 1
            elif true == c:          # it was c, but predicted something else
                fn += 1

        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

        print(f"{c} precision: {precision:.2f} recall: {recall:.2f} f1: {f1:.2f}")

if __name__ == "__main__":
    nbc = NaiveBayesClassifier()
    results = load_small("trainingSet.txt")
    # print(len(results))
    # print(results)

    nbc.train(results)

    # print(nbc.documents)                    # expect {'POS': 6, 'NEU': 6, 'NEG': 6}
    # print(nbc.word_count["NEU"]["software"]) # expect 3
    # print(nbc.total_word_count)
    # print(len(nbc.vocabulary))

    # print(nbc.prior("POS"))
    # print(nbc.likelihood("software", "NEU"))
    # print(nbc.likelihood("software", "POS"))
    # print(nbc.likelihood("zzz", "POS"))

    test_docs = load_small("testSet.txt")
    pairs = evaluate(nbc, test_docs)
    metrics(pairs)


    