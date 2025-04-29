import spacy

nlp = spacy.load("en_core_web_sm")

def extract_tags(text):
    doc = nlp(text)
    tags = [ent.label_ for ent in doc.ents]
    return list(set(tags))
