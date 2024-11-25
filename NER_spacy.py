import spacy

nlp = spacy.load("en_core_web_sm")

def get_ner_types(doc):
    """
    Given a list of words, returns a list of tuples with (word, type),
    limiting types to PER (Person), ORG (Organization), MISC (Miscellaneous), and O (Outside).
    """
    # Create a SpaCy document from the list of words
    text = " ".join(doc)
    spacy_doc = nlp(text)

    # List to store the results (word, type)
    result = []

    # Iterate over the entities found by SpaCy
    for ent in spacy_doc.ents:
        # Check the entity type and map it to the required ones
        if ent.label_ == "PERSON":
            entity_type = "PER"
        elif ent.label_ == "ORG":
            entity_type = "ORG"
        elif ent.label_ == "MISC":
            entity_type = "MISC"
        else:
            entity_type = "O"  # For any other entity type or non-entity words

        # Add the word and its type to the result list
        result.append((ent.text, entity_type))

    # Handle words that are not part of any entity (label them as "O")
    # Create a list of all words with their types, ensuring all words are processed
    for word in spacy_doc:
        # If the word is not part of any entity, mark it as "O"
        if not any(word.text == ent.text for ent in spacy_doc.ents):
            result.append((word.text, "O"))

    return result

