import spacy

def spacy_file_loader(file_path):
    '''load test file data into nested lists, with the inner lists storing sentences splited by -DOCSTART-'''
    doc = []
    i = -1  #counter
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            tokens = line.split()
            word = tokens[0]
            if word == "-DOCSTART-":
                doc.append([])
                i = i + 1
            else:
                doc[i].append(word)
    return doc

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
    MISC_labels = {"LANGUAGE", "NORP"} 
    LOC_labels = {"GPE", "LOC"} 
    for word in spacy_doc:
        if word.ent_type_ == "PERSON":
            entity_type = "PER"
        elif word.ent_type_ == "ORG":
            entity_type = "ORG"
        elif word.ent_type_ in LOC_labels:
            entity_type = "LOC"
        elif word.ent_type_ in MISC_labels:
            entity_type = "MISC"
        else:
            entity_type = "O"  

        # Append the word and its entity type as a tuple
        result.append((word.text, entity_type))

    return result