import re
import sys
import gzip
import json
import math
import time
from pathlib import Path


DATA_PATH = Path(__file__).parent / "data"
SMALL_DATA_PATH = DATA_PATH / "SmallDataset"
LEXICON_PATH = DATA_PATH / "subjectivity_clues_hltemnlp05" / "subjectivity_clues_hltemnlp05" / "subjclueslen1-HLTEMNLP05.tff"
AMAZON_PATH = DATA_PATH / "Video_Games_5.json.gz"

# Extra training documents built to fix "The program sucks." (see report)
FIX_DOCS = [("NEG", "This program sucks.")]
LEXICON_DATA_PATH  = Path(__file__).parent / "data" / "subjectivity_clues_hltemnlp05"
AMAZON_DATA_PATH = Path(__file__).parent / "data"

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
            line = line.strip() # removes leading/trailing whitespace & newlines
            # skips empty lines
            if not line:
                continue
            label, text = line[:3], line[4:] # splits the line into the label and the text
            results.append((label, text))

    return results

def load_lexicon(path=LEXICON_PATH):
    """
    Loads the MPQA Subjectivity Clues lexicon.

    Each line looks like:
    type=weaksubj len=1 word1=abandoned pos1=adj stemmed1=n priorpolarity=negative

    return: dict of word -> "positive" / "negative" (neutral and "both" words are skipped)
    """
    lexicon = {}
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            fields = dict(part.split("=", 1) for part in line.split() if "=" in part)
            word = fields.get("word1")
            polarity = fields.get("priorpolarity")
            if word and polarity in ("positive", "negative"):
                lexicon[word] = polarity
    return lexicon

def star_to_label(stars):
    """Maps an Amazon star rating to a sentiment class: 1-2 NEG, 3 NEU, 4-5 POS."""
    if stars <= 2:
        return "NEG"
    if stars == 3:
        return "NEU"
    return "POS"

def load_amazon(path=AMAZON_PATH, test=False):
    """
    Streams the Amazon reviews one at a time (a generator, so the whole file is never in memory).
    Uses the full review body ("reviewText") as the text.

    Every 5th review is held out as the test set (20%), since no separate Amazon test file was provided.
    - test=False: yields the other 80% (training)
    - test=True:  yields only the held-out 20%

    yields: (label, text)
    """
    with gzip.open(path, 'rt', encoding='utf-8') as file:
        for i, line in enumerate(file):
            if (i % 5 == 0) != test:
                continue
            review = json.loads(line)
            text = review.get("reviewText")
            stars = review.get("overall")
            if not text or stars is None:
                continue
            yield star_to_label(stars), text

def preprocess(text):
    text = text.lower()
    # keep only letters, digits, apostrophes and whitespace (handles any punctuation, not just ?!,.)
    text = re.sub(r"[^a-z0-9'\s]", ' ', text)
    return text.split()

class NaiveBayesClassifier:
    def __init__(self):
        # stores the number of reviews per class
        self.documents = {}
        # how often each word appears in each class
        self.word_count = {}
        # how many words each class has in total, repeats are included
        self.total_word_count = {}
        # every unique word seen in training
        self.vocabulary = set()
        # precomputed log P(c) and log P(w|c), built at the end of training
        self.log_prior = {}
        self.log_likelihood = {}

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

        self.build_log_tables()

    def build_log_tables(self):
        """
        Final training step: computes log P(c) for every class and log P(w|c) for every
        class and every word in the vocabulary (add-1 smoothing), and stores them so
        classifying is just table lookups.
        """
        total_docs = sum(self.documents.values())
        vocab_size = len(self.vocabulary)

        self.log_prior = {}
        self.log_likelihood = {}
        for c in self.documents:
            self.log_prior[c] = math.log(self.documents[c] / total_docs)

            counts = self.word_count.get(c, {})
            denominator = self.total_word_count.get(c, 0) + vocab_size
            self.log_likelihood[c] = {
                word: math.log((counts.get(word, 0) + 1) / denominator)
                for word in self.vocabulary
            }

    def prior(self, c):
        class_docs = self.documents[c]
        total_docs = sum(self.documents.values())
        return class_docs / total_docs

    def likelihood(self, word, c):
        num_in_class = self.word_count[c].get(word, 0) + 1 # La Place smooting
        total_in_class = self.total_word_count[c] + len(self.vocabulary)
        return num_in_class / total_in_class

    def score(self, words, c):
        # log P(c) + sum of log P(w|c), using the tables built during training
        result = self.log_prior[c]
        table = self.log_likelihood[c]

        for word in words:
            # OOV words are not in the table, so they are skipped
            if word in table:
                result += table[word]

        return result

    def classify(self, text):
        classes = self.documents.keys()

        text = preprocess(text)

        ranking = {}

        for c in classes:
            s = self.score(text, c)
            ranking[c] = s

        return max(ranking, key=ranking.get)

    def add_lexicon(self, lexicon):
        """
        Adds the MPQA lexicon as extra training evidence: each positive word gets +1 count
        in POS, each negative word gets +1 count in NEG. Document counts (priors) are unchanged,
        since the lexicon entries are not real reviews.
        """
        polarity_to_class = {"positive": "POS", "negative": "NEG"}

        for word, polarity in lexicon.items():
            c = polarity_to_class[polarity]
            self.word_count.setdefault(c, {})
            self.word_count[c][word] = self.word_count[c].get(word, 0) + 1
            self.total_word_count[c] = self.total_word_count.get(c, 0) + 1
            self.vocabulary.add(word)

        # counts and vocabulary changed, so the log tables must be rebuilt
        self.build_log_tables()

    def explain(self, text):
        """
        Prints everything needed to classify a document by hand:
        the counts, priors, each word's likelihood per class, and the log sums.
        """
        tokens = preprocess(text)
        classes = list(self.documents.keys())

        print(f'\nExplaining: "{text}"')
        print(f"tokens: {tokens}")
        print(f"|V| = {len(self.vocabulary)}")
        for c in classes:
            print(f"{c}: docs = {self.documents[c]}, total words = {self.total_word_count[c]}, "
                  f"denominator = {self.total_word_count[c] + len(self.vocabulary)}, "
                  f"P({c}) = {self.prior(c):.4f}")

        print()
        for word in tokens:
            if word not in self.vocabulary:
                print(f"  {word!r}: OOV, ignored")
                continue
            parts = []
            for c in classes:
                count = self.word_count[c].get(word, 0)
                parts.append(f"{c}: ({count}+1)/{self.total_word_count[c] + len(self.vocabulary)}"
                             f" = {self.likelihood(word, c):.5f}")
            print(f"  {word!r}: " + " | ".join(parts))

        print()
        for c in classes:
            print(f"  log score {c}: {self.score(tokens, c):.4f}")
        print(f"  prediction: {self.classify(text)}")


def evaluate(model:NaiveBayesClassifier, test_docs, verbose=True):

    pairs = []

    for label, text in test_docs:
        prediction = model.classify(text)
        pairs.append((label, prediction))

        if verbose:
            marker = "" if prediction == label else "   <-- WRONG"
            print(f"{text} : {prediction} (true: {label}){marker}")

    return pairs

def metrics(pairs):
    # accuracy
    correct = 0
    for true, pred in pairs:
        if true == pred:
            correct += 1
    print(f"accuracy: {correct / len(pairs):.2f}")

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

def run_small_experiments():
    results = load_small("trainingSet.txt")
    test_docs = load_small("testSet.txt")

    # 1. Baseline
    print("=== 1. Baseline ===")
    nbc = NaiveBayesClassifier()
    nbc.train(results)
    metrics(evaluate(nbc, test_docs))

    # 2. Hand-calculation numbers for the misclassified document
    print("\n=== 2. Explain the misclassified document ===")
    nbc.explain("The program sucks.")

    # 3. Retrain with the fix document(s)
    print("\n=== 3. With fix document(s) ===")
    fixed = NaiveBayesClassifier()
    fixed.train(results + FIX_DOCS)
    metrics(evaluate(fixed, test_docs))
    fixed.explain("The program sucks.")

    # 4. Original training data + MPQA lexicon
    print("\n=== 4. With MPQA lexicon ===")
    lexicon = load_lexicon()
    print(f"lexicon words loaded: {len(lexicon)}")
    with_lexicon = NaiveBayesClassifier()
    with_lexicon.train(results)
    with_lexicon.add_lexicon(lexicon)
    metrics(evaluate(with_lexicon, test_docs))


def run_amazon_experiment():
    print("=== 5. Amazon ===")
    model = NaiveBayesClassifier()

    start = time.perf_counter()
    model.train(load_amazon(test=False))
    train_seconds = time.perf_counter() - start

    print(f"training time: {train_seconds:.1f} s (reading + counting + log likelihood tables)")
    print(f"trained on {sum(model.documents.values())} reviews: {model.documents}")
    print(f"total tokens processed: {sum(model.total_word_count.values())}")
    print(f"tokens per class: {model.total_word_count}")
    print(f"vocabulary size: {len(model.vocabulary)}")
    print(f"log likelihoods computed: {sum(len(t) for t in model.log_likelihood.values())}")

    start = time.perf_counter()
    test_docs = list(load_amazon(test=True))
    pairs = evaluate(model, test_docs, verbose=False)
    test_seconds = time.perf_counter() - start

    print(f"testing on {len(test_docs)} reviews took {test_seconds:.1f} s")
    metrics(pairs)


if __name__ == "__main__":
    # python nb.py         -> small dataset experiments
    # python nb.py amazon  -> Amazon experiment
    if len(sys.argv) > 1 and sys.argv[1] == "amazon":
        run_amazon_experiment()
    else:
        run_small_experiments()


    