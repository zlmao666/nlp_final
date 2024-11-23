import final

def parse_test_data(data):
    ''' returns only each word and its NER type from in a list of tuples and also a tokenized sentence version'''
    result = []
    doc = []
    
    for line in data:
        result.append((line[0], line[3]))
        doc.append(line[0])
        
    return result, doc

def main():

    true_data, doc = parse_test_data(final.load_conll_data("conll2003/test.txt"))
    model, nlp, naive_dict, type_vectors = final.bootstrap()
    
    spacy_prediction = final.get_spacy_prediction() # IMPLEMENT THIS FUNCTION in final.py
    naive_prediction = final.sentence_classifier_all_results(doc, naive_dict, type_vectors, model, nlp)
    
    # To Do: write comparison functions and use them to compare true_data against both predictions (use a3 code)
    
    print(naive_prediction) 

if __name__ == "__main__":
    main()