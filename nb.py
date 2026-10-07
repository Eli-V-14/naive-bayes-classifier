import re
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

class NaivebayesClassifier:
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

if __name__ == "__main__":
    nbc = NaivebayesClassifier()
    results = load_small("trainingSet.txt")
    # print(len(results))
    # print(results)

    nbc.train(results)

    print(nbc.documents)                    # expect {'POS': 6, 'NEU': 6, 'NEG': 6}
    print(nbc.word_count["NEU"]["software"]) # expect 3
    print(nbc.total_word_count)
    print(len(nbc.vocabulary))
