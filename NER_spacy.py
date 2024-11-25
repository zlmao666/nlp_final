import spacy

# Load the English NLP model
nlp = spacy.load("en_core_web_sm")

# Input text for NER
text = "The hydrophobic effect the tendency of nonpolar molecules or parts of molecules to aggregate in aqueous media is central to biomolecular recognition. It now seems that there is no single “hydrophobic effect” 1−4 that adequately describes the partitioning of a small apolar ligand between both (i) an aqueous phase and a nonpolar organic phase (e.g., buffer and octanol), and (ii) bulk aqueous buffer and the active site of a protein (i.e., biomolecular recognition). While the molecular-level mechanisms of hydrophobic effects in biomolecular recognition remain a subject of substantial controversy, it is clear that the water molecules surrounding the apolar ligand and filling the active site of the protein are an important part of these mechanisms.1−10 Clarifying the role of water in the hydrophobic effect in protein−ligand binding would be an important contribution to understanding the fundamental, mechanistic basis of molecular recognition. Resolving this mechanism would, however, still leave a (presumably) related phenomena unresolved: so-called, enthalpy−entropy compensation (H/S compensation)."


# Process the text
doc = nlp(text)

# Extract and print entities
for ent in doc.ents:
    print(ent.text, ent.label_)