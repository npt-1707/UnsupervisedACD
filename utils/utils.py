from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np

def extract_corpus(corpus):
    # Create count vectorizer and fit it to the corpus
    count_vectorizer = CountVectorizer()
    count_matrix = count_vectorizer.fit_transform(corpus)
    
    # Transform count matrix to TF-IDF matrix
    tfidf_transformer = TfidfTransformer()
    tfidf_matrix = tfidf_transformer.fit_transform(count_matrix)
    
    # Get feature names
    feature_names = count_vectorizer.get_feature_names_out()
    
    return count_matrix, tfidf_matrix, feature_names

def get_tf_dict(index, feature_names, count_matrix):
    return {word: tf for word, tf in zip(feature_names, count_matrix.toarray()[index])}

def get_idf_dict(index, feature_names, tfidf_matrix):
    return {word: idf for word, idf in zip(feature_names, tfidf_matrix.toarray()[index])}


def soft_cosine_similarity(sen1, sen2, embedding_model, tf, idf):    
    # Compute word embeddings for each word in the sentences
    embeddings_sen1 = [embedding_model[word] for word in sen1 if word in embedding_model]
    embeddings_sen2 = [embedding_model[word] for word in sen2 if word in embedding_model]
    
    numerator = 0.0
    denominator1 = 0.0
    denominator2 = 0.0
    
    for i in range(len(embeddings_sen1)):
        for j in range(len(embeddings_sen2)):
            # Compute cosine similarity between word embeddings
            cosine_sim = np.dot(embeddings_sen1[i], embeddings_sen2[j]) / (
                np.linalg.norm(embeddings_sen1[i]) * np.linalg.norm(embeddings_sen2[j])
            )
            
            # Retrieve IDF values for the corresponding words
            idf_sen1 = idf[sen1[i]]
            idf_sen2 = idf[sen2[j]]
            
            # Update numerator and denominators
            numerator += cosine_sim * idf_sen1 * idf_sen2
            denominator1 += (cosine_sim * idf_sen1) ** 2
            denominator2 += (cosine_sim * idf_sen2) ** 2
    
    # Calculate final soft cosine similarity
    soft_cos_sim = numerator / (np.sqrt(denominator1) * np.sqrt(denominator2))
    
    return soft_cos_sim
