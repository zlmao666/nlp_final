import NER_naive
import NER_spacy
from collections import defaultdict

def parse_test_data(test_data):
    '''Returns each word and its NER type in a list of tuples, and also a tokenized sentence version.'''
    result = []
    doc = []
    
    for line in test_data:
        word = line[0]
        ner_label = line[3]
        
        # Append the word and its simplified NER label
        result.append((word, ner_label))
        doc.append(word)

    return result, doc

def calculate_pre_metrics(predicted, actual):
    """
    Calculates true positives (TP), false positives (FP), and false negatives (FN) for each entity type.
    """
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    
    total = len(predicted)

    for pred, act in zip(predicted, actual):
        _, type_pred = pred
        _, type_act = act

        if type_pred == type_act:  
            if type_pred != 'O':    # True Positive
               true_positives += 1
            else:                   # True Negative
                true_negatives += 1
        else:
            if type_pred != 'O':    # False Positive
                false_positives += 1
            else:                   # False Negative
                false_negatives += 1
                
    assert true_positives+true_negatives+false_negatives+false_positives == total

    return true_positives, true_negatives, false_positives, false_negatives, total


def evaluate_ner_accuracy(pre_metrics):
    """
    Calculates overall accuracy.
    """   
    tp, tn, fp, fn, t = pre_metrics
    
    return (tp+tn)/t


def evaluate_ner_precision(pre_metrics):
    """
    Calculates precision.
    """
    tp, tn, fp, fn, t = pre_metrics
    
    return tp/(tp+fp)


def evaluate_ner_recall(pre_metrics):
    """
    Calculates recall.
    """
    tp, tn, fp, fn, t = pre_metrics

    return tp/(tp+fn)

def evaluate_ner_f1(precision, recall):
    """
    Calculates F1 score.
    """

    if precision + recall == 0:
        return 0
        
    return 2 * (precision * recall) / (precision + recall)

def print_evaluation(predicted, actual):
    """
    Calculates and prints a formatted table of evaluation metrics (accuracy, precision, recall, F1-score).
    """
    # Calculate metrics
    pre_metrics = calculate_pre_metrics(predicted, actual)
    
    accuracy = evaluate_ner_accuracy(pre_metrics)
    precision = evaluate_ner_precision(pre_metrics)
    recall = evaluate_ner_recall(pre_metrics)
    f1 = evaluate_ner_f1(precision, recall)

    # Print header
    print(f"{'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("-" * 40)
    
    # Print overall metrics
    print(f"{accuracy:<10.4f} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f}")

def main():
    # Parameters for the classifier
    model, nlp, naive_dict, type_vectors = NER_naive.bootstrap()

    # Evaluate the classifier
    test_data = NER_naive.load_conll_data("conll2003/test.txt")
    test_result, doc = parse_test_data(test_data)
    naive_result = NER_naive.sentence_classifier_all_results(doc, naive_dict, type_vectors, model, nlp)
    spacy_result = NER_spacy.get_ner_types("conll2003/test.txt")
    print("Naive NER:")
    print_evaluation(naive_result, test_result)
    print("Spacy NER:")
    print_evaluation(spacy_result, test_result)

    
if __name__ == "__main__":
    main()
