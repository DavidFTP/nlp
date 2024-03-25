from transformers import pipeline  # Importing the pipeline module from the transformers library
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import nltk
import numpy as np
import re  # Importing re module for regular expressions
from nltk.corpus import stopwords
import math
from collections import defaultdict  # Importing defaultdict from collections module for dictionary operations
from sklearn.feature_extraction.text import TfidfVectorizer

# Download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Create a text generation pipeline using GPT-Neo model
generator = pipeline('text-generation', model='EleutherAI/gpt-neo-2.7B')

# Prompts for generating text
prompts = [
    "Machine learning is transforming industries through automation and data insights.",  # Prompt 1
    "Polar bears are endangered due to climate change and habitat loss."  # Prompt 2
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
documents = []  # Initialize an empty list to store preprocessed documents
for prompt in prompts:
    # Generate text based on each prompt
    generated_text = generator(prompt, max_length=200, do_sample=True)[0]['generated_text']

    # Preprocess generated text and add to documents list
    documents.append(preprocess_text(generated_text))

# Step 1: Compute Term Frequency (TF)
vectorizer_tf = TfidfVectorizer(use_idf=False, norm=None)  # Initialize TfidfVectorizer for TF computation
tf_matrix = vectorizer_tf.fit_transform(documents)  # Fit and transform documents to get TF matrix

# Step 2: Compute Inverse Document Frequency (IDF)
vectorizer_idf = TfidfVectorizer(use_idf=True, norm=None)  # Initialize TfidfVectorizer for IDF computation
idf_matrix = vectorizer_idf.fit_transform(documents)  # Fit and transform documents to get IDF matrix

# Step 3: Multiply TF by IDF
tfidf_matrix = tf_matrix.multiply(idf_matrix)  # Multiply TF matrix by IDF matrix element-wise

# Step 4: Normalize TF-IDF
vectorizer_tfidf = TfidfVectorizer(use_idf=True, norm='l2')  # Initialize TfidfVectorizer for normalized TF-IDF
tfidf_matrix_normalized = vectorizer_tfidf.fit_transform(
    documents)  # Fit and transform documents to get normalized TF-IDF matrix

# Display TF-IDF for each word in each document
feature_names = vectorizer_tfidf.get_feature_names_out()  # Get feature names (words)
for i, doc in enumerate(documents):  # Iterate over documents
    print(f"Document {i + 1}:")  # Print document number
    for j, word in enumerate(feature_names):  # Iterate over words
        print(f"\t{word}: {tfidf_matrix_normalized[i, j]}")  # Print word and its TF-IDF score for the current document


# Function to compute TF-IDF from scratch
def calculate_tf_idf(documents):
    tf_idf_scores = defaultdict(dict)  # Initialize defaultdict to store TF-IDF scores for each document

    # Calculate IDF for all words
    document_count = len(documents)  # Get total number of documents
    idf_scores = {}  # Initialize dictionary to store IDF scores for each word
    for doc in documents:  # Iterate over documents
        words_set = set(doc.split())  # Get unique words in the current document
        for word in words_set:  # Iterate over words
            if word not in idf_scores:  # Check if IDF score for the word has not been computed yet
                word_doc_count = sum(word in d for d in documents)  # Count documents containing the word
                idf_scores[word] = math.log10(document_count / (1 + word_doc_count)) + 1 # Compute IDF score

    # Calculate TF-IDF for each document
    for doc_id, document in enumerate(documents):  # Iterate over documents
        words = document.split()  # Tokenize document into words
        total_words = len(words)  # Get total number of words in the document
        word_count = defaultdict(int)  # Initialize defaultdict to store word counts
        for word in words:  # Iterate over words
            word_count[word] += 1  # Count occurrences of each word

        tfidf_vector = defaultdict(float)  # Initialize defaultdict to store TF-IDF scores for each word
        for word, count in word_count.items():  # Iterate over word counts
            if word in idf_scores:  # Check if IDF score for the word has been computed
                tf = count / total_words  # Compute TF (Term Frequency)
                tfidf = tf * idf_scores[word]  # Compute TF-IDF score
                tfidf_vector[word] = tfidf  # Store TF-IDF score for the word

        # Normalize TF-IDF vector
        norm = math.sqrt(sum(tfidf ** 2 for tfidf in tfidf_vector.values()))  # Compute L2 norm of TF-IDF vector
        if norm != 0:  # Check if norm is not zero to avoid division by zero
            for word in tfidf_vector:  # Iterate over words
                tf_idf_scores[doc_id][word] = tfidf_vector[word] / norm  # Normalize TF-IDF score and store
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
