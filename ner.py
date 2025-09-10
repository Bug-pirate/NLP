import spacy
from spacy import displacy

nlp = spacy.load("en_core_web_sm")

text = """
Amazon has opened a new headquarters in New York City, creating over 5,000 jobs. 
At the same time, Facebook announced a $500 million investment in renewable energy projects in California. 
The Prime Minister of Canada, Justin Trudeau, met with President Joe Biden in Washington to discuss climate change policies. 
Meanwhile, SpaceX successfully launched its Starship rocket from Texas last Friday.
"""


doc = nlp(text)

# Print entities
print("----- Named Entities -----")
for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)

# Force displacy to use localhost instead of 0.0.0.0
displacy.serve(doc, style="ent", host="127.0.0.1", port=5000)
