import spacy
import gensim.downloader
import numpy as np
import time
import collections

def load_conll_data(file_path):
    attributes = []  

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()  

            # Skip lines that contain -DOCSTART- or are empty
            if line.startswith("-DOCSTART-") or not line:
                continue

            # Split the line into components (word, POS tag, chunk tag, NER tag)
            tokens = line.split()
            word = tokens[0]   
            pos_tag = tokens[1]  
            chunk_tag = tokens[2]  
            ner_tag = map_conll_types(tokens[3])

            # Add the attributes to the list
            attributes.append((word, pos_tag, chunk_tag, ner_tag))

    return attributes

def get_conll_types():
    '''helper function to return a list of types'''
    # types = ('B-ORG', 'O', 'B-MISC', 'B-PER', 'I-PER', 'B-LOC', 'I-ORG', 'I-MISC', 'I-LOC')
    types = ('ORG', 'O', 'MISC', 'PER', 'LOC')
    return types
    
def map_conll_types(ner_type):
    if ner_type == 'O':
        return ner_type
    else:
        return ner_type[2:]

def build_naive_dict(lst):
    ''' builds naive dictionary, a nested dictionary containing raw counts of each type for each word '''
    naive_dict = collections.defaultdict(lambda: dict.fromkeys(get_conll_types(), 0))
    for ls in lst:
        word = ls[0].lower()
        ner_type = ls[3]
        (naive_dict[word])[ner_type] += 1

    return naive_dict
    
def probabalize_naive_dict(naive_dict):
    ''' converts raw counts into probabilities in naive dictionary '''
    for key in naive_dict:
        nested_dict = naive_dict[key]
        total = sum(nested_dict.values())
        for ner_type, val in nested_dict.items():
            nested_dict[ner_type] = val / total
            
    return naive_dict
    
def classify_conll_types(lst):
    ''' returns a dictionary of type: list of words within the type'''
    dic = dict.fromkeys(get_conll_types(), [])

    for ls in lst:
        word = ls[0].lower()
        ner_type = ls[3]
        dic[ner_type].append(word) # Does not avoid duplcates
    
    ''' OLD CODE
    # Then append the words to the correct types
    for ls in lst:
        for key in dic.keys():
            if ls[3] == key:
                if ls[0] not in dic[key]:       # Avoid replicates
                    dic[key].append(ls[0])     # Append the word to the type
    '''
    return dic

def calculate_category_vectors(type_dict, model):
    ''' returns a dictionary mapping from each type to the mean vector for all of its associated words '''
    type_vectors = dict.fromkeys(type_dict)

    # Loop over all types in the type_dict (NER categories)
    for ner_type, words in type_dict.items():
        # Calculate the average vector for the words in this NER category
        type_vectors[ner_type] = np.mean([model.get_vector(w) for w in words if w in model], axis=0)
        
    return type_vectors
    
def vector_checker(word, type_vectors, model):
    ''' checks word using gensim model vector similarity '''
    best_type = None
    best_similarity = -1  # Cosine similarity ranges from -1 to 1
    
    try:
        word_vec = model.get_vector(word)
    except KeyError:
        # If the word is not in the model's vocabulary,
        if word.isalpha():
            best_type = 'PER'
        else:
            best_type = 'O'
        return best_type

    
    # Loop over all types in the type_dict (NER categories)
    for ner_type in type_vectors:
        
        type_vector = type_vectors[ner_type]
 
        if type_vector is None or np.isnan(type_vector).any():
            continue


        # Calculate cosine similarity between the word vector and the average category vector
        similarity = np.dot(word_vec, type_vector) / (np.linalg.norm(word_vec) * np.linalg.norm(type_vector))


        # Update the best similarity and corresponding type if necessary
        if similarity > best_similarity:
            best_similarity = similarity
            best_type = ner_type

    return best_type

def naive_checker(word, naive_dict):
    ''' checks word naively against training data and returns highest probability category, returns None if not found'''
    
    if word not in naive_dict:
        return None
        
    ner_types = naive_dict[word]
    best_type = max(ner_types.keys(), key=ner_types.get)
    return best_type
    
def word_classifier(word, naive_dict, type_vectors, model):
    ''' returns the type of the word'''
    word = word.lower()

    best_type = naive_checker(word, naive_dict)
    
    if best_type == None:
        # print(f"'{word}' was not found naively, using vectors")
        best_type = vector_checker(word, type_vectors, model)
    
    # print(best_type,"is the best type for", word)
    return best_type
    
def sentence_classifier(doc, naive_dict, type_vectors, model, nlp):
    ''' classifies a sentence and returns only interesting results '''
    tokenized = None
    if type(doc) is str:
        tokenized = nlp(doc)
    
    interesting_results = []
    
    if tokenized:
        for token in tokenized:
            best_type = word_classifier(token.text, naive_dict, type_vectors, model)
            if best_type not in ('O', 'B-MISC', 'I-MISC', 'MISC'):
                interesting_results.append((token.text, best_type))
    else:
        for token in doc:
            best_type = word_classifier(token, naive_dict, type_vectors, model)
            if best_type not in ('O', 'B-MISC', 'I-MISC', 'MISC'):
                interesting_results.append((token, best_type))
    
    return interesting_results
    
def sentence_classifier_all_results(doc, naive_dict, type_vectors, model, nlp):
    ''' classifies a sentence and returns all results '''
    tokenized = None
    if type(doc) is str:
        tokenized = nlp(doc)
    
    results = []
    
    if tokenized:
        for token in tokenized:
            best_type = word_classifier(token.text, naive_dict, type_vectors, model)
            results.append((token.text, best_type))
    else:
        for token in doc:
            best_type = word_classifier(token, naive_dict, type_vectors, model)
            results.append((token, best_type))

    
    return results
    
def get_spacy_prediction(): #IMPLEMENT THIS

    return None
    
def bootstrap():
    ''' load models and data '''
    print('Loading model(s)...')
    start = time.time()
    # model = gensim.downloader.load('word2vec-google-news-300') # This one is slower to load
    model = gensim.downloader.load('glove-twitter-100')
    # model = gensim.downloader.load('glove-wiki-gigaword-50') # This one is faster to load
    nlp = spacy.load("en_core_web_sm")
    print('Done. ({} seconds)'.format(time.time() - start))
    print('-------------')
    print('Building data...')
    start = time.time()
    data = load_conll_data("conll2003/train.txt")
    naive_dict = probabalize_naive_dict(build_naive_dict(data))
    type_dict = classify_conll_types(data)
    type_vectors = calculate_category_vectors(type_dict, model)
    print('Done. ({} seconds)'.format(time.time() - start))
    print('-------------')
    
    return model, nlp, naive_dict, type_vectors

def main():

    model, nlp, naive_dict, type_vectors = bootstrap()
    
    # word_classifier("napoleon", naive_dict, type_vectors, model)
    # word_classifier("fork", naive_dict, type_vectors, model)
    
    # sentence = "Ryan Reynolds paid $10,000 of his own money for the right to wear a shirt with The Golden Girl’s Bea Arthur on it in Deadpool. The estate agreed for a donation in that amount to a charity of their choosing. Ryan paid it himself because he felt you couldn’t have Deadpool without Bea Arthur."
    # type_sentence = sentence_classifier(sentence, naive_dict, type_vectors, model, nlp)
    
    with open('testfile.txt', 'r') as file:
        text = file.read().replace('\n', '')

    print(sentence_classifier(text, naive_dict, type_vectors, model, nlp))
    

if __name__ == "__main__":
    main()