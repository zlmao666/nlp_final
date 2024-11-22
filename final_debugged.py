import spacy
import gensim.downloader
import numpy as np
import time

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
            ner_tag = tokens[3]  

            # Add the attributes to the list
            attributes.append([word, pos_tag, chunk_tag, ner_tag])

    return attributes

def get_conll_typs(list):
    '''helper function to return a list of types'''
    types = []
    for ls in list:
        if ls[3] not in types:
            types.append(ls[3])

def classify_conll_types(list):
    ''' returns a dictionary of type: list of words within the type'''
    dic = {}
    #First get all types, make the value as an empty string to store the words
    for ls in list:
        if ls[3] not in dic:
            dic[ls[3]] = []

    # Then append the words to the correct types
    for ls in list:
        for key in dic.keys():
            if ls[3] == key:
                if ls[0] not in dic[key]:       # Avoid replicates
                    dic[key].append(ls[0])     # Append the word to the type
    return dic


def word_classifier(word, type_dict, model):
    ''' returns the type of the word'''
    word = word.lower()
    try:
        # Get the vector for the word
        word_vec = model.get_vector(word)
    except KeyError:
        # If the word is not in the model's vocabulary, the word is classified as other (O)
        print(f"Word '{word}' not in vocabulary, classifying as O.")
        return "O"

    print(f"Vector for word '{word}': {word_vec[:5]}...")  # Show the first 5 values of the vector

    best_type = None
    best_similarity = -1  # Cosine similarity ranges from -1 to 1

    # Loop over all types in the type_dict (NER categories)
    for ner_type, words in type_dict.items():
        # Calculate the average vector for the words in this NER category
        avg_type_vec = np.mean([model.get_vector(w) for w in words if w in model], axis=0)

        if avg_type_vec is None or np.isnan(avg_type_vec).any():
            print(f"Skipping type {ner_type} due to invalid vector (NaN or None).")
            continue

        print(f"Avg vector for type '{ner_type}': {avg_type_vec[:5]}...")  # Show first 5 elements of the avg vector

        # Calculate cosine similarity between the word vector and the average category vector
        similarity = np.dot(word_vec, avg_type_vec) / (np.linalg.norm(word_vec) * np.linalg.norm(avg_type_vec))

        print(f"Similarity between '{word}' and type '{ner_type}': {similarity:.4f}")

        # Update the best similarity and corresponding type if necessary
        if similarity > best_similarity:
            best_similarity = similarity
            best_type = ner_type

    return best_type




def main():
    print('Loading model...')
    start = time.time()
    # model = gensim.downloader.load('word2vec-google-news-300') # This one is slower to load
    # model2 = gensim.downloader.load('glove-twitter-50')

    model = gensim.downloader.load('glove-wiki-gigaword-50') # This one is faster to load
    print('Done. ({} seconds)'.format(time.time() - start))


    data = load_conll_data("conll2003/train.txt")
    type_dict = classify_conll_types(data)
    type1 = word_classifier("henry",type_dict, model)
    print(type1)

if __name__ == "__main__":
    main()