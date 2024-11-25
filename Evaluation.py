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
            simp_type = ner_label[2:]  
        elif ner_label.startswith("I-"):
            simp_type = ner_label[2:]  
        else:
            simp_type = ner_label  
        
        # Append the word and its simplified NER label
        result.append((word, simp_type))
        doc.append(word)

    return result, doc

def get_naive_result(doc, naive_dict, type_vectors, model):
    result = []
    for word in doc:        
        type_predicted = NER_naive.word_classifier(word, naive_dict, type_vectors, model)
        result.append((word,type_predicted))

    return result

def calculate_tp_fp_fn(predicted, actual):
    """
    Calculates true positives (TP), false positives (FP), and false negatives (FN) for each entity type.
    """
    true_positives = defaultdict(int)
    false_positives = defaultdict(int)
    false_negatives = defaultdict(int)

    for pred, act in zip(predicted, actual):
        _, type_pred = pred
        _, type_act = act

        if type_pred == type_act and type_pred != 'O':  # True Positive
            true_positives[type_pred] += 1
        elif type_pred != type_act:
            if type_pred != 'O':  # False Positive
                false_positives[type_pred] += 1
            if type_act != 'O':  # False Negative
                false_negatives[type_act] += 1

    return true_positives, false_positives, false_negatives


def evaluate_ner_accuracy(predicted, actual):
    """
    Calculates overall accuracy by comparing the second item of two lists of tuples (word, type).
    """
    correct = sum(1 for pred, act in zip(predicted, actual) if pred[1] == act[1])
    total = len(predicted)
    return correct / total if total > 0 else 0.0


def evaluate_ner_precision(predicted, actual):
    """
    Calculates precision for each entity type and overall precision.
    """
    true_positives, false_positives, _ = calculate_tp_fp_fn(predicted, actual)
    precision = {}

    for entity in true_positives:
        tp = true_positives[entity]
        fp = false_positives[entity]
        precision[entity] = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    avg_precision = sum(precision.values()) / len(precision) if precision else 0.0
    return precision, avg_precision


def evaluate_ner_recall(predicted, actual):
    """
    Calculates recall for each entity type and overall recall.
    """
    true_positives, _, false_negatives = calculate_tp_fp_fn(predicted, actual)
    recall = {}

    for entity in true_positives:
        tp = true_positives[entity]
        fn = false_negatives[entity]
        recall[entity] = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    avg_recall = sum(recall.values()) / len(recall) if recall else 0.0
    return recall, avg_recall


def evaluate_ner_f1(precision, recall):
    """
    Calculates F1 score for each entity type and overall F1 score.
    """
    f1_scores = {}

    for entity in precision:
        p = precision[entity]
        r = recall.get(entity, 0.0)
        f1_scores[entity] = 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0

    avg_f1 = sum(f1_scores.values()) / len(f1_scores) if f1_scores else 0.0
    return f1_scores, avg_f1


def print_evaluation(predicted, actual):
    """
    Calculates and prints a formatted table of evaluation metrics (accuracy, precision, recall, F1-score).
    """
    # Calculate metrics
    accuracy = evaluate_ner_accuracy(predicted, actual)
    precision, avg_precision = evaluate_ner_precision(predicted, actual)
    recall, avg_recall = evaluate_ner_recall(predicted, actual)
    f1, avg_f1 = evaluate_ner_f1(precision, recall)

    # Print header
    print(f"{'Entity':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("-" * 40)

    # Print per-entity metrics
    for entity in precision:
        print(f"{entity:<10} {precision[entity]:<10.4f} {recall[entity]:<10.4f} {f1[entity]:<10.4f}")

    # Print overall metrics
    print("-" * 40)
    print(f"{'Overall':<10} {avg_precision:<10.4f} {avg_recall:<10.4f} {avg_f1:<10.4f}")
    print(f"{'Accuracy':<10} {accuracy:<10.4f}")

def main():
    # Parameters for the classifier
    train_data = load_test_data("conll2003/train.txt")
    naive_dict = NER_naive.probabalize_naive_dict(NER_naive.build_naive_dict(train_data))
    type_dict = NER_naive.classify_conll_types(train_data)
    model = gensim.downloader.load('glove-wiki-gigaword-50')
    type_vectors = NER_naive.calculate_category_vectors(type_dict, model)
    

    # Evaluate the classifier
    test_data = load_test_data("conll2003/test.txt")
    test_result, doc = parse_test_data(test_data)
    doc = doc[:500]
    naive_result = get_naive_result(doc, naive_dict, type_vectors, model)
    spacy_result = NER_spacy.get_ner_types("conll2003/test.txt")
    print("Naive NER:")
    print_evaluation(naive_result, test_result)
    print("Spacy NER:")
    print_evaluation(spacy_result, test_result)

    
if __name__ == "__main__":
    main()
