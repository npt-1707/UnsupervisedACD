from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

try:
    english_stopwords = stopwords.words('english')
except:
    import nltk
    nltk.download('stopwords')
    english_stopwords = stopwords.words('english')
    
class Preprocessor:
    def __init__(self):
        self.stop_words = english_stopwords
        self.tokenizer = word_tokenize
        
    def preprocess(self, sentence):
        # Tokenize the sentence
        tokens = self.tokenizer(sentence)
        # Convert tokens to lowercase
        tokens = [token.lower() for token in tokens]
        # Remove stopwords and punctuation
        tokens = [token for token in tokens if token.isalnum()
                    and token not in self.stop_words]
        # Join tokens back into a sentence
        preprocessed_sentence = ' '.join(tokens)
        return preprocessed_sentence