import sys
from collections import Counter, OrderedDict
import itertools
from itertools import islice, count, groupby
import pandas as pd
import os
import re
from operator import itemgetter
from time import time
from pathlib import Path
import pickle
from google.cloud import storage
from collections import defaultdict
from contextlib import closing
import numpy as np

PROJECT_ID = 'ir-project-415515'
bucket_name = 'irproj_26051997'


def get_bucket(bucket_name):
    return storage.Client(PROJECT_ID).bucket(bucket_name)


def _open(path, mode, bucket=None):
    if bucket is None:
        return open(path, mode)
    return bucket.blob(path).open(mode)


# Let's start with a small block size of 30 bytes just to test things out.
BLOCK_SIZE = 1999998


class MultiFileWriter:
    """ Sequential binary writer to multiple files of up to BLOCK_SIZE each. """

    def __init__(self, base_dir, name, bucket_name=None):
        self._base_dir = Path(base_dir)
        self._name = name
        self._bucket = None if bucket_name is None else get_bucket(bucket_name)
        self._file_gen = (_open(str(self._base_dir / f'{name}_{i:03}.bin'),
                                'wb', self._bucket)
                          for i in itertools.count())
        self._f = next(self._file_gen)

    def write(self, b):
        locs = []
        while len(b) > 0:
            pos = self._f.tell()
            remaining = BLOCK_SIZE - pos
            # if the current file is full, close and open a new one.
            if remaining == 0:
                self._f.close()
                self._f = next(self._file_gen)
                pos, remaining = 0, BLOCK_SIZE
            self._f.write(b[:remaining])
            name = self._f.name if hasattr(self._f, 'name') else self._f._blob.name
            locs.append((name, pos))
            b = b[remaining:]
        return locs

    def close(self):
        self._f.close()


class MultiFileReader:
    """ Sequential binary reader of multiple files of up to BLOCK_SIZE each. """

    def __init__(self, base_dir, bucket_name=None):
        self._base_dir = Path(base_dir)
        self._bucket = None if bucket_name is None else get_bucket(bucket_name)
        self._open_files = {}

    def read(self, locs, n_bytes):
        b = []
        for f_name, offset in locs:             
#             f_name = str(self._base_dir / f_name)
            if f_name not in self._open_files:
                self._open_files[f_name] = _open(f_name, 'rb', self._bucket)
            f = self._open_files[f_name]
            f.seek(offset)
            n_read = min(n_bytes, BLOCK_SIZE - offset)
            b.append(f.read(n_read))
            n_bytes -= n_read
        return b''.join(b)

    def close(self):
        for f in self._open_files.values():
            f.close()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


TUPLE_SIZE = 6  # We're going to pack the doc_id and tf values in this
# many bytes.
TF_MASK = 2 ** 16 - 1  # Masking the 16 low bits of an integer


class InvertedIndex:
    def __init__(self, base_dir, docs={}, filter_tf=False):
        """ Initializes the inverted index and add documents to it (if provided).
        Parameters:
        -----------
          docs: dict mapping doc_id to list of tokens
        """
        self.base_dir = base_dir
        # stores document frequency per term
        self.df = Counter()
        # stores total frequency per term
        self.term_total = Counter()
        # stores posting list per term while building the index (internally),
        # otherwise too big to store in memory.
        self._posting_list = defaultdict(list)
        # mapping a term to posting file locations, which is a list of
        # (file_name, offset) pairs. Since posting lists are big we are going to
        # write them to disk and just save their location in this list. We are
        # using the MultiFileWriter helper class to write fixed-size files and store
        # for each term/posting list its list of locations. The offset represents
        # the number of bytes from the beginning of the file where the posting list
        # starts.
        self.posting_locs = defaultdict(list)
        # stores the length of each document
        self.doc_lengths = defaultdict(float)
        self.num_docs = 0  # Total number of documents
        self.avg_doc_lengths = 0

        for doc_id, tokens in docs.items():
            self.add_doc(doc_id, tokens, filter_tf)

    
    def add_doc(self, doc_id, tokens, filter_tf):
        """ Adds a document to the index with a given `doc_id` and tokens.
        It counts the term frequency, updates the index, and calculates
        the document length.
        """
        # Update term frequency count and document length
        w2cnt = Counter(tokens)
        self.term_total.update(w2cnt)
        doc_length = sum(w2cnt.values())
        self.doc_lengths[doc_id] = doc_length
        self.num_docs += 1

        # Update document frequency and posting list
        for token, (id, count) in tokens:
            self.df[token] = self.df.get(token, 0) + 1
            self._posting_list[token].append((id, count))

    def write_index(self, base_dir, name, bucket_name=None):
        """ Write the in-memory index to disk. Results in the file: 
            (1) `name`.pkl containing the global term stats (e.g. df).
            (2) `name`_tf_idf.pkl containing the tf_idf values.
        """
        #### GLOBAL DICTIONARIES ####
        self._write_globals(base_dir, name, bucket_name)
        #### TF-IDF VALUES ####
#         self._write_tf_idf(base_dir, name, bucket_name)

    def _write_globals(self, base_dir, name, bucket_name):
        path = str(Path(base_dir) / f'{name}.pkl')
        bucket = None if bucket_name is None else get_bucket(bucket_name)
        with _open(path, 'wb', bucket) as f:
            pickle.dump(self, f)

    def _write_tf_idf(self, base_dir, name, bucket_name):
        tf_idf_dict = {}
        for doc_id in self.doc_lengths.keys():
            tf_idf_dict[doc_id] = self.calculate_tf_idf(doc_id)
        path = str(Path(base_dir) / f'{name}_tf_idf.pkl')
        bucket = None if bucket_name is None else get_bucket(bucket_name)
        with _open(path, 'wb', bucket) as f:
            pickle.dump(tf_idf_dict, f)

    def __getstate__(self):
        """ Modify how the object is pickled by removing the internal posting lists
            from the object's state dictionary. 
        """
        state = self.__dict__.copy()
        del state['_posting_list']
        return state

    def posting_lists_iter(self, bucket_name=None):
        """ A generator that reads one posting list from disk and yields 
            a (word:str, [(doc_id:int, tf:int), ...]) tuple.
        """
        with closing(MultiFileReader(self.base_dir, bucket_name)) as reader:
            for w, locs in self.posting_locs.items():
                b = reader.read(locs, self.df[w] * TUPLE_SIZE)
                posting_list = []
                for i in range(self.df[w]):
                    doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
                    tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
                    posting_list.append((doc_id, tf))
                yield w, posting_list

    def read_a_posting_list(self, w, bucket_name=None):
        posting_list = []
        if not w in self.posting_locs:
            return posting_list
        with closing(MultiFileReader(self.base_dir, bucket_name)) as reader:
            locs = self.posting_locs[w]
            b = reader.read(locs, self.df[w] * TUPLE_SIZE)
            for i in range(self.df[w]):
                doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
                tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
                posting_list.append((doc_id, tf))
        return posting_list

    def calculate_idf(self):
        """Calculates and returns IDF values for all terms."""
        idf = {}
        for term, df in self.df.items():
            idf[term] = np.log(self.num_docs / (df + 1e-4))
        return idf

    def calculate_tf_idf(self, doc_id, bucket_name=None):
        """Calculates and returns TF-IDF vector for a document."""
        tf_idf = {}
        doc_length = self.doc_lengths[doc_id]

        # Iterate over terms in the document's posting list
        for term, posting_list in self.posting_locs.items():
            # Calculate TF-IDF for the term in the document
            for doc_id_in_pl, tf_in_pl in self.read_a_posting_list(term, bucket_name):
                if doc_id_in_pl == doc_id:
                    tf_idf[term] = (tf_in_pl / doc_length) * self.idf.get(term, 0)

        return tf_idf
    
    def calculate_tf_idf_for_query(self, tokens):
        """Calculates and returns TF-IDF vector for a query."""
        query_terms = Counter(tokens)
        query_length = len(query_terms)
        query_tf_idf = {}
        for term, count in query_terms.items():
            if term in self.df:
                tf = count / query_length
                # Apply Laplace smoothing
                smoothed_idf = self.idf.get(term, 0)  # Default to 0 if term not found in idf
                query_tf_idf[term] = tf * smoothed_idf
            else:
                # Term not found in corpus, handle smoothing
                # You can choose any smoothing method here, such as setting a default idf value
                query_tf_idf[term] = 0  # For simplicity, setting TF-IDF score to 0 for unseen terms
        return query_tf_idf


    @staticmethod
    def write_a_posting_list(b_w_pl, base_dir, bucket_name=None):
        posting_locs = defaultdict(list)
        bucket_id, list_w_pl = b_w_pl

        with closing(MultiFileWriter(base_dir, bucket_id, bucket_name)) as writer:
            for w, pl in list_w_pl:
                # convert to bytes
                b = b''.join([(doc_id << 16 | (tf & TF_MASK)).to_bytes(TUPLE_SIZE, 'big')
                              for doc_id, tf in pl])
                # write to file(s)
                locs = writer.write(b)
                # save file locations to index
                posting_locs[w].extend(locs)
            path = str(Path(base_dir) / f'{bucket_id}_posting_locs.pickle')
            bucket = None if bucket_name is None else get_bucket(bucket_name)
            with _open(path, 'wb', bucket) as f:
                pickle.dump(posting_locs, f)
        return bucket_id

    @staticmethod
    def read_index(base_dir, name, bucket_name=None):
        path = str(Path(base_dir) / f'{name}.pkl')
        bucket = None if bucket_name is None else get_bucket(bucket_name)
        with _open(path, 'rb', bucket) as f:
            return pickle.load(f)

    @staticmethod
    def read_tf_idf(base_dir, name, bucket_name=None):
        path = str(Path(base_dir) / f'{name}_tf_idf.pkl')
        bucket = None if bucket_name is None else get_bucket(bucket_name)
        with _open(path, 'rb', bucket) as f:
            return pickle.load(f)
