import os
import re
import sys
import pickle
import hashlib
from time import time
from collections import Counter, defaultdict
from itertools import groupby
from operator import itemgetter
from pathlib import Path

import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import *

import subprocess
from google.cloud import storage
import subprocess

import math
import numpy as np
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import heapq

subprocess.call(['pip', 'install', 'pyspark'])
subprocess.call(['pip', 'install', 'google-cloud-storage'])

# Download NLTK stopwords
nltk.download('stopwords')

# Constants
BUCKET_NAME = 'irproj_26051997'
TEXT_INDEX_PATH = 'indices/text_index/postings_text_gcp/text_index.pkl'
TITLE_INDEX_PATH = 'indices/title_index/postings_title_gcp/title_index.pkl'
dict_of_titles_PATH = 'title_id_dict/title_id_dict.pkl'
pagerank_PATH = "Page_Rank_dict.pkl"


# Load index files
def read_pickle(bucket_name, pickle_route):
    """
    Read a pickle file from Google Cloud Storage.

    Args:
    - bucket_name (str): Name of the bucket
    - pickle_route (str): Path to the pickle file

    Returns:
    - object: Deserialized object from the pickle file
    """
    client = storage.Client()
    blob = client.bucket(bucket_name).blob(pickle_route)
    pick = pickle.loads(blob.download_as_string())
    return pick


text_index = read_pickle(BUCKET_NAME, TEXT_INDEX_PATH)
title_index = read_pickle(BUCKET_NAME, TITLE_INDEX_PATH)
dict_of_titles_per_doc = read_pickle(BUCKET_NAME, dict_of_titles_PATH)
pagerank = read_pickle(BUCKET_NAME, pagerank_PATH)

# Tokenizer
english_stopwords = set(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became"]

all_stopwords = english_stopwords.union(corpus_stopwords)
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)


def tokenize(text, expand=False):
    """
    Tokenize the input text.

    Args:
    - text (str): Input text to be tokenized

    Returns:
    - list: List of tokens
    """
    return [token.group() for token in RE_WORD.finditer(text.lower()) if token.group() not in all_stopwords]


def calc_idf(query_tokens, inverted_index):
    """
    Calculate the IDF (Inverse Document Frequency) for query tokens.

    Args:
    - query_tokens (list): List of query tokens
    - inverted_index (InvertedIndex): Inverted index object

    Returns:
    - dict: Dictionary containing IDF values for each token
    """
    idf = defaultdict(float)
    N = inverted_index.num_docs

    def calculate_idf(term):
        if term in inverted_index.df.keys():
            n = inverted_index.df.get(term, 0)
            return (term, np.log(1 + ((N - n + 1e-4) / (n + 1e-4))))
        return None

    with ThreadPoolExecutor(max_workers=len(query_tokens)) as executor:
        results = executor.map(calculate_idf, query_tokens)

    for result in results:
        if result:
            term, value = result
            idf[term] = value
    return idf


def bm25_for_term(term, inverted_index, idf, doc_lengths, avg_doc_length, k1=1.5, b=0.75):
    """
    Calculates bm25 score per term
    """
    term_bm25_dict = defaultdict(float)
    if term in inverted_index.df:
        pl = inverted_index.read_a_posting_list(term, BUCKET_NAME)
        for doc_id, tf in pl:
            doc_length = doc_lengths[doc_id]
            term_score = (idf[term] * tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_length / avg_doc_length)))
            term_bm25_dict[doc_id] += term_score
    return term_bm25_dict


def merge_dicts(dicts):
    merged_dict = defaultdict(float)
    for d in dicts:
        for key, value in d.items():
            merged_dict[key] += value
    return merged_dict


def bm25(query_tokens, inverted_index, k1=1.5, b=0.75):
    """
    Calculate BM25 scores for query tokens.

    Args:
    - query_tokens (list): List of query tokens
    - inverted_index (InvertedIndex): Inverted index object
    - k1 (float): BM25 parameter
    - b (float): BM25 parameter

    Returns:
    - dict: Dictionary containing BM25 scores for each document
    """
    idf = calc_idf(query_tokens, inverted_index)
    doc_lengths = inverted_index.doc_lengths
    avg_doc_length = inverted_index.avg_doc_length

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(bm25_for_term, term, inverted_index, idf, doc_lengths, avg_doc_length, k1, b) for
                   term in query_tokens]

    term_bm25_dicts = [future.result() for future in as_completed(futures)]
    bm25_dict = merge_dicts(term_bm25_dicts)

    return bm25_dict


def get_title(doc_id):
    """
    inputs = doc_id
    output = the title of doc_id
    """
    title = dict_of_titles_per_doc.get(doc_id)
    if title:
        return title
    return "No match"


def merge_all(text_scores, title_scores, text_w, title_w, pr_w, k=30):
    top_scores_heap = []  # Heap to keep track of top-k scoring documents
    min_score = float('inf')

    # Iterate over documents with non-zero scores in any of the dictionaries
    relevant_docs = set(text_scores.keys()) | set(title_scores.keys())  # | set(anchor_scores.keys())
    for doc_id in relevant_docs:
        # Calculate total score for the document

        total_score = (text_scores.get(doc_id, 0) * text_w +
                       title_scores.get(doc_id, 0) * title_w +
                       np.log(pagerank.get(doc_id, [0])[0] + 1e-4) * pr_w)

        if total_score < min_score:
            min_score = total_score

        # If heap size exceeds k, pop the smallest (negative) element
        if len(top_scores_heap) < k:
            # Add negated score and doc_id to the heap
            heapq.heappush(top_scores_heap, (total_score, doc_id))
        else:
            # Add negated score and doc_id to the heap
            heapq.heappushpop(top_scores_heap, (total_score, doc_id))

    minmax = top_scores_heap[0][0] - min_score

    # Negate scores back to positive and reverse the heap order
    max_scores = [(doc_id, (score - min_score) / minmax) for score, doc_id in top_scores_heap]

    return max_scores

def _search(query):
    """
    Perform a search operation for the given query.

    Args:
    - query (str): Search query

    Returns:
    - list: List of tuples containing document IDs and corresponding titles
    """
    query_tokens = tokenize(query)  # Tokenize once and reuse

    with ThreadPoolExecutor(max_workers=3) as executor:
        future_text_scores = executor.submit(bm25, query_tokens, text_index)
        future_title_scores = executor.submit(bm25, query_tokens, title_index)

        text_scores = future_text_scores.result()
        title_scores = future_title_scores.result()

    if len(query_tokens) <= title_index.avg_doc_length:
        title_w = 0.75
        text_w = 0.25
    else:
        title_w = 0.5
        text_w = title_w
    pr_w = 1

    max_scores = merge_all(text_scores, title_scores, text_w, title_w, pr_w)

    final_result = [(str(doc_id), get_title(doc_id)) for doc_id, score in max_scores]

    return final_result
