from transformers import pipeline  
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import nltk
import numpy as np
import re  # re module for regular expressions
from nltk.corpus import stopwords
import math
from collections import defaultdict  # defaultdict from collections module for dictionary operations
from sklearn.feature_extraction.text import TfidfVectorizer

# Download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Create a text generation pipeline using GPT-Neo model
generator = pipeline('text-generation', model='EleutherAI/gpt-neo-2.7B')

# Prompts for generating text
prompts = [
    "Machine learning is transforming industries through automation and data insights.", 
    "Polar bears are endangered due to climate change and habitat loss."  
]


# Preprocess the text
def preprocess_text(text):
    # Clean data by removing symbols and special characters
    cleaned_text = re.sub(r'[^a-zA-Z0-9_\s]', '', text)

    # Normalize text to lowercase
    cleaned_text = cleaned_text.lower()

    # Tokenize the text into words
    words_list = word_tokenize(cleaned_text)

    # Lemmatize each word to its base form
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words_list]

    # Get English stop words
    stop_words = set(stopwords.words('english'))

    # Remove stop words from the text
    filtered_words = [word for word in lemmatized_words if word not in stop_words]

    # Get unique words from the list
    unique_words = set(filtered_words)

    return ' '.join(unique_words)


# Preprocess and collect documents
documents = []  
for prompt in prompts:
    generated_text = generator(prompt, max_length=200, do_sample=True)[0]['generated_text']
    documents.append(preprocess_text(generated_text))

# Compute Term Frequency (TF)
vectorizer_tf = TfidfVectorizer(use_idf=False, norm=None)  
tf_matrix = vectorizer_tf.fit_transform(documents)  

# Compute Inverse Document Frequency (IDF)
vectorizer_idf = TfidfVectorizer(use_idf=True, norm=None)  
idf_matrix = vectorizer_idf.fit_transform(documents)  

# Multiply TF by IDF
tfidf_matrix = tf_matrix.multiply(idf_matrix)  

#  Normalize TF-IDF
vectorizer_tfidf = TfidfVectorizer(use_idf=True, norm='l2')  
tfidf_matrix_normalized = vectorizer_tfidf.fit_transform(documents)  

# Display TF-IDF for each word in each document
feature_names = vectorizer_tfidf.get_feature_names_out()  
for i, doc in enumerate(documents):  
    print(f"Document {i + 1}:")  
    for j, word in enumerate(feature_names):  
        print(f"\t{word}: {tfidf_matrix_normalized[i, j]}") 


# TF-IDF from scratch
def calculate_tf_idf(documents):
    tf_idf_scores = defaultdict(dict)  # Initialize defaultdict to store TF-IDF scores for each document
    document_count = len(documents)  
    idf_scores = {}  # Initialize dictionary to store IDF scores for each word
    for doc in documents: 
        words_set = set(doc.split())  # Get unique words in the current document
        for word in words_set:  
            if word not in idf_scores:  # Check if IDF score for the word has not been computed yet
                word_doc_count = sum(word in d for d in documents)  # Count documents containing the word
                idf_scores[word] = math.log10(document_count + 1 / (1 + word_doc_count)) + 1 

    # Calculate TF-IDF for each document
    for doc_id, document in enumerate(documents):  
        words = document.split()  
        total_words = len(words)  
        word_count = defaultdict(int)  
        for word in words: 
            word_count[word] += 1  # Count occurrences of each word

        tfidf_vector = defaultdict(float)  
        for word, count in word_count.items(): 
            if word in idf_scores:  
                tf = count / total_words  
                tfidf = tf * idf_scores[word]  
                tfidf_vector[word] = tfidf  

        # Normalize TF-IDF vector
        norm = math.sqrt(sum(tfidf ** 2 for tfidf in tfidf_vector.values()))  # Compute L2 norm of TF-IDF vector
        if norm != 0:  # Check if norm is not zero to avoid division by zero
            for word in tfidf_vector:  
                tf_idf_scores[doc_id][word] = tfidf_vector[word] / norm  
        else:
            tf_idf_scores[doc_id] = tfidf_vector  # If norm is zero, store unnormalized TF-IDF
    return tf_idf_scores

# Calculate TF-IDF from scratch
tfidf_matrix = calculate_tf_idf(documents)


# Print TF-IDF
for i, (doc_id, tfidf_doc) in enumerate(tfidf_matrix.items(), start=1):
    print(f"Document {doc_id}:")
    for word, tfidf_score in tfidf_doc.items():
        print(f"{word}: {tfidf_score}")
    print("-" * 50)
