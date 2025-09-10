# Step One: Import nltk and download necessary packages
import nltk

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')  # needed in new NLTK
nltk.download('maxent_ne_chunker')
nltk.download('maxent_ne_chunker_tab')           # ✅ NEW
nltk.download('words')

# Step Two: Load Data
sentence = """In the aftermath of several cases of misconduct by New York police officers during the 1990s, Loretta E. Lynch, who was the chief federal prosecutor in Brooklyn, spoke strongly about the deep sense of broken trust felt by African-Americans. She emphasized that law enforcement carried the responsibility to mend the long history of miscommunication and mistrust."""

# Step Three: Tokenise, find parts of speech and chunk words
for sent in nltk.sent_tokenize(sentence):
    words = nltk.word_tokenize(sent)
    pos_tags = nltk.pos_tag(words)
    chunks = nltk.ne_chunk(pos_tags)
    for chunk in chunks:
        if hasattr(chunk, 'label'):
            print(chunk.label(), ' '.join(c[0] for c in chunk))
