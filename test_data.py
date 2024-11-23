import final
import compare
import numpy as np
import sys

def parse_test_data(data):
    ''' returns only each word and its NER type from in a list of tuples and also a tokenized sentence version'''
    result = []
    doc = []
    
    for line in data:
        result.append((line[0], line[3]))
        doc.append(line[0])
        
    return result, doc
    
def create_comparison_vectors(data):
    ''' returns a numpy array of all results '''
    
    result = np.array([x[1] for x in data])
    return result
    
def print_data(accuracy, precision, recall, f1):
    
    try:
        print("Accuracy: ", accuracy)
        print("Precision: ", precision)
        print("Recall: ", recall)
        print("f1 Score: ", f1)
    except NameError:
        print("Trouble printing data", file=sys.stderr)

def main():

    data, doc = parse_test_data(final.load_conll_data("conll2003/test.txt"))
    model, nlp, naive_dict, type_vectors = final.bootstrap()
    
    # spacy_prediction = create_comparison_vectors(final.get_spacy_prediction()) # IMPLEMENT FUNCTION get_spacy_prediction() in final.py
    naive_prediction = create_comparison_vectors(final.sentence_classifier_all_results(doc, naive_dict, type_vectors, model, nlp))
    true_data = create_comparison_vectors(data)
    
    print(naive_prediction.shape)
    print(true_data.shape)
    
    # print("----- Expected vs. Naive Prediction -----")
    # print_data(compare.get_metrics(true_data, naive_prediction))
    
    # To Do: write comparison functions and use them to compare true_data against both predictions (use a3 code)
    

if __name__ == "__main__":
    main()