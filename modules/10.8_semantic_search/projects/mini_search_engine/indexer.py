# -*- coding: utf-8 -*-
# indexer.py
#
# Module 10.8 -- Semantic Search Systems
# Project: Mini Search Engine
#
# PURPOSE:
#   The Indexer loads documents, encodes them to vectors, and builds
#   a searchable index. It also saves and loads the index from disk.
#
#   This represents the OFFLINE PHASE of the search pipeline:
#   - Run once when you add/update documents
#   - Results saved to disk
#   - Never run during search (only load the saved index)
#
# C# ANALOGY:
#   This is like a DocumentRepository class with a BuildIndex() method
#   and SaveToFile() / LoadFromFile() methods using serialization.

import numpy as np    # For vector math
import pickle         # For saving/loading Python objects to disk (like BinaryFormatter in C#)
import os             # For file path operations


# ============================================================
# WORD VECTOR TABLE
# ============================================================
# 10-dimensional vectors representing word meaning across topics:
# [science, history, tech, sports, geography, people, nature, society, art, misc]

WORD_VECS = {
    # Science
    "gravity":        np.array([0.9, 0.0, 0.1, 0.0, 0.0, 0.1, 0.1, 0.0, 0.0, 0.0]),
    "quantum":        np.array([0.9, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "physics":        np.array([0.9, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "chemistry":      np.array([0.9, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0]),
    "biology":        np.array([0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0]),
    "evolution":      np.array([0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0]),
    "dna":            np.array([0.9, 0.0, 0.1, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0]),
    "photosynthesis": np.array([0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.0, 0.0, 0.0]),
    "atoms":          np.array([0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "energy":         np.array([0.8, 0.0, 0.2, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0]),
    "space":          np.array([0.8, 0.0, 0.2, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "planet":         np.array([0.8, 0.0, 0.1, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "black":          np.array([0.3, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.0]),
    "hole":           np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "star":           np.array([0.6, 0.0, 0.0, 0.0, 0.2, 0.3, 0.0, 0.0, 0.1, 0.0]),
    "relativity":     np.array([0.9, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0]),
    "universe":       np.array([0.8, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.1, 0.0]),
    "climate":        np.array([0.6, 0.0, 0.0, 0.0, 0.3, 0.0, 0.5, 0.2, 0.0, 0.0]),
    "ecosystem":      np.array([0.6, 0.0, 0.0, 0.0, 0.3, 0.0, 0.7, 0.0, 0.0, 0.0]),
    "oxygen":         np.array([0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0]),
    # History
    "war":            np.array([0.0, 0.9, 0.0, 0.0, 0.0, 0.2, 0.0, 0.3, 0.0, 0.0]),
    "ancient":        np.array([0.0, 0.9, 0.0, 0.0, 0.2, 0.0, 0.0, 0.1, 0.1, 0.0]),
    "empire":         np.array([0.0, 0.9, 0.0, 0.0, 0.2, 0.2, 0.0, 0.3, 0.0, 0.0]),
    "civilization":   np.array([0.0, 0.9, 0.1, 0.0, 0.2, 0.2, 0.0, 0.3, 0.2, 0.0]),
    "roman":          np.array([0.0, 0.9, 0.0, 0.0, 0.2, 0.2, 0.0, 0.2, 0.2, 0.0]),
    "egypt":          np.array([0.0, 0.8, 0.0, 0.0, 0.5, 0.0, 0.1, 0.1, 0.2, 0.0]),
    "medieval":       np.array([0.0, 0.9, 0.0, 0.0, 0.1, 0.1, 0.0, 0.2, 0.2, 0.0]),
    "revolution":     np.array([0.0, 0.8, 0.1, 0.0, 0.0, 0.2, 0.0, 0.5, 0.0, 0.0]),
    "king":           np.array([0.0, 0.7, 0.0, 0.0, 0.0, 0.5, 0.0, 0.4, 0.0, 0.0]),
    "battle":         np.array([0.0, 0.8, 0.0, 0.2, 0.0, 0.2, 0.0, 0.3, 0.0, 0.0]),
    "dynasty":        np.array([0.0, 0.9, 0.0, 0.0, 0.1, 0.3, 0.0, 0.2, 0.1, 0.0]),
    "slavery":        np.array([0.0, 0.8, 0.0, 0.0, 0.0, 0.2, 0.0, 0.7, 0.0, 0.0]),
    "silk":           np.array([0.0, 0.7, 0.0, 0.0, 0.5, 0.0, 0.0, 0.3, 0.1, 0.0]),
    "plague":         np.array([0.3, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0]),
    # Technology
    "computer":       np.array([0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0]),
    "internet":       np.array([0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0]),
    "software":       np.array([0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0]),
    "artificial":     np.array([0.1, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0]),
    "intelligence":   np.array([0.2, 0.0, 0.8, 0.0, 0.0, 0.2, 0.0, 0.1, 0.0, 0.0]),
    "robot":          np.array([0.1, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0]),
    "algorithm":      np.array([0.2, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "programming":    np.array([0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0]),
    "semiconductor":  np.array([0.2, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "satellite":      np.array([0.3, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "blockchain":     np.array([0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0]),
    "electricity":    np.array([0.4, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    # Sports
    "football":       np.array([0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0]),
    "tennis":         np.array([0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "olympic":        np.array([0.0, 0.1, 0.0, 0.9, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0]),
    "swimming":       np.array([0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "athlete":        np.array([0.0, 0.0, 0.0, 0.9, 0.0, 0.2, 0.0, 0.1, 0.0, 0.0]),
    "championship":   np.array([0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0]),
    "basketball":     np.array([0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "cricket":        np.array([0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "marathon":       np.array([0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    # Geography
    "ocean":          np.array([0.2, 0.0, 0.0, 0.0, 0.9, 0.0, 0.5, 0.0, 0.0, 0.0]),
    "mountain":       np.array([0.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.4, 0.0, 0.0, 0.0]),
    "river":          np.array([0.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.4, 0.0, 0.0, 0.0]),
    "desert":         np.array([0.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.3, 0.0, 0.0, 0.0]),
    "rainforest":     np.array([0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.7, 0.0, 0.0, 0.0]),
    "volcano":        np.array([0.4, 0.0, 0.0, 0.0, 0.8, 0.0, 0.2, 0.0, 0.0, 0.0]),
    "continent":      np.array([0.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "coral":          np.array([0.2, 0.0, 0.0, 0.0, 0.8, 0.0, 0.6, 0.0, 0.0, 0.0]),
    "glacier":        np.array([0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.3, 0.0, 0.0, 0.0]),
    "amazon":         np.array([0.0, 0.0, 0.2, 0.0, 0.7, 0.0, 0.5, 0.0, 0.0, 0.0]),
    # Common words (low signal)
    "the":  np.zeros(10), "a":    np.zeros(10), "and": np.zeros(10),
    "of":   np.zeros(10), "in":   np.zeros(10), "is":  np.zeros(10),
    "to":   np.zeros(10), "that": np.zeros(10), "for": np.zeros(10),
    "with": np.zeros(10), "it":   np.zeros(10), "by":  np.zeros(10),
    "was":  np.zeros(10), "are":  np.zeros(10), "has": np.zeros(10),
    "as":   np.zeros(10), "an":   np.zeros(10), "be":  np.zeros(10),
    "from": np.zeros(10), "which":np.zeros(10), "or":  np.zeros(10),
    "on":   np.zeros(10), "this": np.zeros(10), "at":  np.zeros(10),
}


def embed(text):
    """
    Convert text to a vector by averaging word vectors.

    Parameters:
        text (str): Any text to encode

    Returns:
        numpy array: 10-dimensional vector
    """
    words = text.lower().split()               # Split into words
    vecs = []
    for word in words:
        vec = WORD_VECS.get(word)              # Look up word vector
        if vec is not None:
            vecs.append(vec)
    if not vecs:
        return np.zeros(10)                    # Unknown text: zero vector
    return np.mean(np.stack(vecs), axis=0)    # Average word vectors


class Indexer:
    """
    Indexes a collection of documents for the search engine.

    Responsibilities:
    1. Accept a list of {"id", "title", "text"} document dicts
    2. Encode each document to a vector using embed()
    3. Build a flat vector index (vectors + IDs)
    4. Save the index to disk using pickle
    5. Load an existing index from disk

    In C# terms: this is like a SearchIndexBuilder class with
    Serialize() and Deserialize() methods.
    """

    def __init__(self):
        """Initialize empty Indexer."""
        self.documents = {}        # {doc_id: {"title": str, "text": str}}
        self.vectors = {}          # {doc_id: numpy_array}
        self.doc_ids = []          # Ordered list of doc IDs (for fast iteration)

    def load_documents(self, docs):
        """
        Load a list of documents into the indexer.

        Parameters:
            docs (list): List of {"id": str, "title": str, "text": str} dicts
        """
        print(f"[Indexer] Loading {len(docs)} documents...")

        for doc in docs:
            doc_id = doc["id"]          # Extract document ID
            title = doc["title"]        # Extract title
            text = doc["text"]          # Extract body text

            # Store the document content for later retrieval
            self.documents[doc_id] = {
                "title": title,
                "text":  text,
                "full":  title + " " + text,   # Combined for encoding
            }
            self.doc_ids.append(doc_id)     # Add to ordered list

        print(f"[Indexer] Loaded {len(self.documents)} documents.")

    def build_index(self):
        """
        Encode all documents and build the vector index.
        This is the OFFLINE step -- called once when documents are loaded.

        For each document:
        - Combine title + text for a richer representation
        - Encode to a 10-dim vector using embed()
        - Store in self.vectors
        """
        print(f"[Indexer] Building vector index for {len(self.documents)} docs...")

        for doc_id, doc_data in self.documents.items():

            # Encode the full document (title + text combined)
            # Using title + text gives more context than text alone
            combined_text = doc_data["full"]    # "title text body..."
            vector = embed(combined_text)       # Convert to vector

            # Store the vector with its document ID
            self.vectors[doc_id] = vector

        print(f"[Indexer] Index built. {len(self.vectors)} document vectors ready.")

    def save(self, path):
        """
        Save the index to disk using pickle.
        Pickle serializes Python objects to binary files.

        In C# terms: BinaryFormatter.Serialize() or JsonSerializer.Serialize()

        Parameters:
            path (str): File path to save to (e.g., "search_index.pkl")
        """
        # Prepare all data for saving
        index_data = {
            "documents": self.documents,    # Original doc text
            "vectors":   self.vectors,      # Encoded vectors
            "doc_ids":   self.doc_ids,      # Ordered ID list
        }

        # Write to file using pickle (binary format)
        with open(path, "wb") as f:        # "wb" = write binary
            pickle.dump(index_data, f)     # Serialize Python object to binary

        print(f"[Indexer] Index saved to: {path}")
        print(f"          ({os.path.getsize(path) / 1024:.1f} KB)")

    def load(self, path):
        """
        Load a previously saved index from disk.

        Parameters:
            path (str): File path to load from

        Returns:
            bool: True if loaded successfully, False if file not found
        """
        if not os.path.exists(path):        # Check if file exists first
            print(f"[Indexer] No saved index found at: {path}")
            return False

        # Read from file using pickle
        with open(path, "rb") as f:        # "rb" = read binary
            index_data = pickle.load(f)    # Deserialize binary to Python object

        # Restore all data
        self.documents = index_data["documents"]
        self.vectors   = index_data["vectors"]
        self.doc_ids   = index_data["doc_ids"]

        print(f"[Indexer] Loaded index from: {path}")
        print(f"          {len(self.documents)} documents, {len(self.vectors)} vectors")
        return True

    def get_document(self, doc_id):
        """
        Retrieve document content by ID.

        Parameters:
            doc_id (str): Document identifier

        Returns:
            dict: {"title": str, "text": str} or None if not found
        """
        return self.documents.get(doc_id)
