# -*- coding: utf-8 -*-
# reranker.py
#
# Module 10.8 -- Semantic Search Systems
# Project: Mini Search Engine
#
# PURPOSE:
#   The Reranker takes the Searcher's top-K candidates and re-scores them
#   more accurately by looking at the INTERACTION between query and document.
#
#   This is the CROSS-ENCODER step of the pipeline:
#   - Accurate (sees query + doc together)
#   - Slower (cannot pre-compute, must run at query time per pair)
#   - Takes top-20 from Searcher, returns refined top-5
#
# C# ANALOGY:
#   This is like a RelevanceScorer class with a Score(query, doc) method.
#   It is called after initial retrieval, on a small set of candidates.

import numpy as np    # For any numerical operations


class Reranker:
    """
    Cross-encoder style re-ranker.

    Takes top-k candidates from the Searcher and re-ranks them by
    computing a more accurate relevance score for each (query, doc) pair.

    Unlike the Searcher which compares pre-encoded vectors,
    the Reranker always sees the original query and document TEXT together.
    This allows it to detect:
    - Exact phrase matches ("black hole" as a phrase)
    - High overlap between query and document topics
    - Whether the document DIRECTLY answers the query

    Usage:
        reranker = Reranker()
        final_results = reranker.rerank(
            query="how does gravity work",
            candidates=[("blackhole_01", 0.82), ...],
            indexer=indexer,
            top_n=5
        )
        # Returns: [("blackhole_01", 0.91, "Black hole"), ...]
    """

    def __init__(self):
        """Initialize the Reranker. No model loading needed for our simulation."""
        pass   # Nothing to initialize for the rule-based cross-encoder

    def _tokenize(self, text):
        """
        Simple tokenizer: lowercase, split on spaces, remove punctuation.

        Parameters:
            text (str): Any text to tokenize

        Returns:
            set: Set of unique lowercase words (punctuation stripped)
        """
        words = text.lower().split()   # Split on whitespace
        clean = set()
        for word in words:
            # Strip common punctuation from start/end of each word
            word = word.strip(".,!?;:\"'()[]{}--")
            if word:                   # Only add non-empty strings
                clean.add(word)
        return clean

    def _cross_score(self, query, title, text):
        """
        Compute a relevance score for a (query, document) pair.
        This simulates what a real cross-encoder would do.

        A real cross-encoder:
        - Concatenates [CLS] query [SEP] document [SEP]
        - Runs through full transformer (query + doc tokens interact via attention)
        - Outputs a 0-1 relevance score

        Our simulation uses three signals:
        1. Token overlap: what fraction of query words are in the document?
        2. Title relevance: does the document title directly mention query topics?
        3. Phrase matching: does the document contain 2-word phrases from the query?

        Parameters:
            query (str): The user's search query
            title (str): Document title
            text  (str): Document body text

        Returns:
            float: Relevance score from 0.0 (irrelevant) to 1.0 (perfect match)
        """
        # Tokenize all inputs
        query_tokens = self._tokenize(query)       # Set of query words
        title_tokens = self._tokenize(title)        # Set of title words
        doc_tokens   = self._tokenize(text)         # Set of body text words
        all_doc_tokens = title_tokens | doc_tokens  # Union: all words in doc

        if not query_tokens:                        # Guard against empty query
            return 0.0

        # --- Signal 1: Token overlap ---
        # How many query words appear anywhere in the document (title + text)?
        shared = query_tokens.intersection(all_doc_tokens)
        overlap_ratio = len(shared) / len(query_tokens)    # 0.0 to 1.0

        # --- Signal 2: Title relevance ---
        # If query words appear in the TITLE, the document is very likely directly relevant
        # Titles are more informative per word than body text
        title_shared = query_tokens.intersection(title_tokens)
        title_ratio = len(title_shared) / max(len(title_tokens), 1)   # 0.0 to 1.0

        # --- Signal 3: Phrase matching ---
        # If 2-word query phrases appear verbatim in the document,
        # the document is very specifically addressing the query topic
        phrase_bonus = 0.0
        query_words_list = query.lower().split()    # Ordered list for phrase building
        doc_full_lower = (title + " " + text).lower()   # Combined doc text

        for i in range(len(query_words_list) - 1):         # Build 2-word phrases
            phrase = query_words_list[i] + " " + query_words_list[i+1]
            if phrase in doc_full_lower:                   # Does the phrase appear?
                phrase_bonus += 0.12                        # Reward each matched phrase

        # Bonus for 3-word phrases (even stronger signal)
        for i in range(len(query_words_list) - 2):
            phrase3 = " ".join(query_words_list[i:i+3])   # 3-word phrase
            if phrase3 in doc_full_lower:
                phrase_bonus += 0.08                        # Additional bonus

        # --- Combine all signals ---
        # Weights: 40% overlap + 20% title + 40% phrase
        # (phrase matching is a strong signal of specificity)
        raw_score = (0.40 * overlap_ratio +
                     0.20 * title_ratio +
                     phrase_bonus)

        # Clamp to [0.0, 1.0] -- cannot exceed 1.0 or go below 0.0
        score = min(1.0, max(0.0, raw_score))
        return score

    def rerank(self, query, candidates, indexer, top_n=5):
        """
        Re-rank candidates by computing accurate (query, doc) relevance scores.

        Parameters:
            query      (str):     The user's search query
            candidates (list):    List of (doc_id, bi_score) from the Searcher
            indexer    (Indexer): Reference to the Indexer for looking up doc text
            top_n      (int):     How many final results to return

        Returns:
            List of (doc_id, cross_score, title, snippet) tuples,
            sorted by cross_score descending (best first).

            Each tuple contains:
            - doc_id:      str  -- the document ID
            - cross_score: float -- the cross-encoder relevance score
            - title:       str  -- the document title
            - snippet:     str  -- first 150 chars of document text
        """
        reranked = []

        for doc_id, bi_score in candidates:

            # Look up the original document text from the indexer
            doc_data = indexer.get_document(doc_id)

            if doc_data is None:               # Skip if doc not found
                continue

            title = doc_data.get("title", "")   # Get title
            text  = doc_data.get("text",  "")   # Get body text

            # Compute cross-encoder score for this (query, doc) pair
            # This is the key step -- seeing query and doc TOGETHER
            cross_score = self._cross_score(query, title, text)

            # Create a short snippet: first 150 characters of text
            snippet = text[:150] + "..." if len(text) > 150 else text

            # Store result: (doc_id, score, title, snippet)
            reranked.append((doc_id, cross_score, title, snippet))

        # Sort by cross-encoder score (best first)
        # This is the re-ranking step -- original bi-encoder order is discarded
        reranked.sort(key=lambda x: x[1], reverse=True)

        # Return only top_n results
        return reranked[:top_n]
