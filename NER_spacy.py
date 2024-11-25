import spacy
from collections import Counter

def spacy_file_loader(file_path):
    '''load test file data into nested lists, with the inner lists storing sentences as splited by -DOCSTART-'''
    doc = []
    i = -1  #counter
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
        
            line = line.strip()
            if not line:
                continue

            tokens = line.split()
            word = tokens[0]
            if word == "-DOCSTART-":
                doc.append([])
                i = i + 1
            else:
                doc[i].append(word)
    return doc

def get_most_frequent(ent_types):

    c = Counter(ent_types)
    
    return c.most_common(1)[0][0]

def map_spacy_types(entity_type):

    MISC_labels = {"LANGUAGE", "NORP"} 
    LOC_labels = {"GPE", "LOC"} 

    if entity_type == "PERSON":
        entity_type = "PER"
    elif entity_type == "ORG":
        entity_type = "ORG"
    elif entity_type in LOC_labels:
        entity_type = "LOC"
    elif entity_type in MISC_labels:
        entity_type = "MISC"
    else:
        entity_type = "O"
    
    return entity_type
    
def process_spacy_doc(sentence_ls):
    """
    Helper function to process a single document (list of lines) using SpaCy.
    Returns a list of tuples (word, type).
    """
    nlp = spacy.load("en_core_web_sm")
    # Join the document lines into a single string for SpaCy processing
    text = " ".join(sentence_ls)
    spacy_doc = nlp(text)

    result = []
    i = 0
    j = 0
    
    while j < len(spacy_doc):
        # print(j)
        word = spacy_doc[j]
        text = word.text
        jj = 0
        ent_types = [word.ent_type_]
        
        while text != sentence_ls[i]:
            # print(text, " vs ", sentence_ls[i])
            jj += 1
            text += spacy_doc[j+jj].text
            ent_types.append(spacy_doc[j+jj].ent_type_)
        
        # print(text, " vs ", sentence_ls[i])
        ent_type = get_most_frequent(ent_types)
        
        result.append((text, map_spacy_types(ent_type)))
        j += jj + 1
        i += 1

    return result

def get_ner_types(file_path):
    """
    Returns a clean list of tuples (word, type), limiting types to 
    PER (Person), ORG (Organization), MISC (Miscellaneous), and O (Outside).
    """
    result = []
    doc = spacy_file_loader(file_path)
    for sentence_ls in doc:
        result.extend(process_spacy_doc(sentence_ls))

    return result

def main():
    print(get_ner_types("conll2003/test copy.txt"))
    
if __name__ == "__main__":
    main()