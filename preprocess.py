
# Import nltk modules for pre-processing

import nltk # Import Natural Language Toolkit module
import re # Import regular expression Module


stopwords = set(nltk.corpus.stopwords.words('english'))
stopwords.update(['911','like', 'name', 'okay',
                'ok', 'coming', 'could', 'days', 'everyone',
                'get', 'give', 'going', 'liked', 'say', 'th',
                'still', 'vs','call','operator',
                'phone','hello','building'])

stemmer = nltk.stem.SnowballStemmer('english')
token_pattern = r"(?u)\b\w\w+\b"

# Function for stemming

def stem_tokens(tokens, stemmer):
    stemmed = []
    for token in tokens:
        stemmed.append(stemmer.stem(token))
    return stemmed

# Wrapped function for tokenization, Regular_expression, Stop words removal and stemming 
def preprocess_transcripts(transcript,
                    token_pattern=token_pattern,
                    exclude_stopword=True,
                    stem=True):
    token_pattern = re.compile(token_pattern, flags = 0 )
    tokens = [x.lower() for x in token_pattern.findall(transcript)]
    tokens_stemmed = tokens
    if stem:
        tokens_stemmed = stem_tokens(tokens, stemmer)
    if exclude_stopword:
        tokens_stemmed = [x for x in tokens_stemmed if x not in stopwords]

    return tokens_stemmed


'''Reference: Bird, Steven, Edward Loper and Ewan Klein (2009), Natural Language Processing with Python. O’Reilly Media Inc.'''

