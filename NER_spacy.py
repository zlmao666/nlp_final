import spacy

nlp = spacy.load("en_core_web_sm")

def spacy_file_loader(file_path):
    doc = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            tokens = line.split()
            doc.append(tokens[0])
    return doc

def get_ner_types(file_path):
    """
    Returns a clean list of tuples (word, type), limiting types to 
    PER (Person), ORG (Organization), MISC (Miscellaneous), and O (Outside).
    """
    result = []
    current_doc = []
    doc = spacy_file_loader(file_path)
    for line in doc:
        # Check for "-DOCSTART-" to indicate a new document
        if line.strip() == "-DOCSTART-":
            if current_doc:
                result.extend(process_spacy_doc(current_doc))
                current_doc = []  # Reset for the next document
        else:
            # Add non-empty lines to the current document
            if line.strip():
                current_doc.append(line.strip())

    # Process the last document if any
    if current_doc:
        result.extend(process_spacy_doc(current_doc))

    return result


def process_spacy_doc(doc_lines):
    """
    Helper function to process a single document (list of lines) using SpaCy.
    Returns a list of tuples (word, type).
    """
    # Join the document lines into a single string for SpaCy processing
    text = " ".join(doc_lines)
    spacy_doc = nlp(text)

    # List to store results
    result = []
    MISC_labels = {"WORK_OF_ART", "PRODUCT", "EVENT", "LANGUAGE", "NORP"} 
    LOC_labels = {"GPE", "LOC","FAC"} 
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
            entity_type = "O"  # For non-entity words

        # Append the word and its entity type as a tuple
        result.append((word.text, entity_type))

    return result
