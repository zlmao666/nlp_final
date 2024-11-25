import NER_naive
import NER_spacy
import gensim.downloader
from collections import defaultdict

def load_test_data(test_file):
    return NER_naive.load_conll_data(test_file)

def parse_test_data(test_data):
    '''Returns each word and its NER type in a list of tuples, and also a tokenized sentence version.'''
    result = []
    doc = []
    
    for line in test_data:
        word = line[0]
        ner_label = line[3]

        # Strip the prefixes ('B-' or 'I-') from the NER label if they exist
        if ner_label.startswith("B-"):
            simp_type = ner_label[2:]  # Remove 'B-' prefix
        elif ner_label.startswith("I-"):
            simp_type = ner_label[2:]  # Remove 'I-' prefix
        else:
            simp_type = ner_label  # No prefix, so use the original label
        
        # Append the word and its simplified NER label
        result.append((word, simp_type))
        doc.append(word)

    return result, doc

def get_naive_result(doc, naive_dict, type_dict, model):
    result = []
    for word in doc:        
        type_predicted = NER_naive.word_classifier(word, naive_dict, type_dict, model)
        result.append((word,type_predicted))

    return result

# Helper function to calculate True Positives and False Negatives
def calculate_tp_fn(predicted, actual):
    '''Calculates true positives (TP) and false negatives (FN) for each entity type.'''
    true_positives = defaultdict(int)
    false_negatives = defaultdict(int)

    for pred, act in zip(predicted, actual):
        _, type_pred = pred
        _, type_act = act

        if type_pred == type_act:  # True Positive (correct prediction)
            if type_pred != 'O':  # Ignore 'O' (Outside)
                true_positives[type_pred] += 1
        else:  # False Negative (misclassifications)
            if type_act != 'O':  # Ignore 'O' (Outside)
                false_negatives[type_act] += 1

    return true_positives, false_negatives


def evaluate_ner_accuracy(predicted, actual):
    '''Compares the second item of two lists of tuples (word, type) and calculates accuracy.'''
    total = 0
    correct = 0
    for pred, act in zip(predicted, actual):
        _, type_pred = pred
        _, type_act = act
        if type_pred == type_act:
            correct += 1
        total += 1
    return correct / total if total > 0 else 0.0


def evaluate_ner_precision(predicted, actual):
    '''Compares the second item of two lists of tuples (word, type) and calculates precision.'''
    true_positives, false_positives = defaultdict(int), defaultdict(int)
    
    # Calculate True Positives and False Negatives
    true_positives, _ = calculate_tp_fn(predicted, actual)

    # Calculate False Positives (predictions that don't match the actual)
    for pred, act in zip(predicted, actual):
        _, type_pred = pred
        _, type_act = act
        if type_pred != type_act and type_pred != 'O':  # False Positive (incorrect prediction)
            false_positives[type_pred] += 1

    precision = {}
    for entity in true_positives:
        tp = true_positives[entity]
        fp = false_positives[entity]
        precision[entity] = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    avg_precision = sum(precision.values()) / len(precision) if precision else 0.0
    return precision, avg_precision


def evaluate_ner_recall(predicted, actual):
    '''Compares the second item of two lists of tuples (word, type) and calculates recall.'''
    true_positives, false_negatives = defaultdict(int), defaultdict(int)
    
    # Calculate True Positives and False Negatives
    true_positives, false_negatives = calculate_tp_fn(predicted, actual)

    recall = {}
    for entity in true_positives:
        tp = true_positives[entity]
        fn = false_negatives[entity]
        recall[entity] = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    avg_recall = sum(recall.values()) / len(recall) if recall else 0.0
    return recall, avg_recall


def evaluate_ner_f1(precision, recall):
    '''Calculates F1 score based on precision and recall.'''
    f1_scores = {}
    for entity in precision:
        p = precision[entity]
        r = recall.get(entity, 0.0)
        f1_scores[entity] = 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0

    avg_f1 = sum(f1_scores.values()) / len(f1_scores) if f1_scores else 0.0
    return f1_scores, avg_f1


def main():
    # Parameters for the classifier
    train_data = load_test_data("conll2003/train.txt")
    naive_dict = NER_naive.probabalize_naive_dict(NER_naive.build_naive_dict(train_data))
    type_dict = NER_naive.classify_conll_types(train_data)
    model = gensim.downloader.load('glove-wiki-gigaword-50')

    # Evaluate the classifier
    test_data = load_test_data("conll2003/test.txt")
    test_result, doc = parse_test_data(test_data)
    doc = doc[:500]    
    naive_result = get_naive_result(doc, naive_dict, type_dict, model)
    #spacy_result = NER_spacy.get_ner_types(doc)
    print(evaluate_ner_accuracy(naive_result, test_result))
    #print(evaluate_ner_accuracy(spacy_result, test_result))

    
if __name__ == "__main__":
    main()
