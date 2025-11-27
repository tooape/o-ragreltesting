"""
BM25 keyword search for O-RAG.

Uses rank-bm25 for efficient BM25 scoring with customizable tokenization.
Designed to work with the hierarchical chunker output.
"""

import re
from typing import List, Dict, Tuple, Optional, Union
import numpy as np

try:
    from rank_bm25 import BM25Okapi, BM25Plus, BM25L
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False


class BM25Searcher:
    """BM25 keyword search with configurable tokenization.

    Supports multiple BM25 variants:
    - Okapi (default): Classic BM25 with k1 and b parameters
    - Plus: BM25+ with lower bound on term frequency contribution
    - L: BM25L with length normalization
    """

    # Default stopwords for Obsidian vault search
    DEFAULT_STOPWORDS = {
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
        "be", "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "can", "this", "that", "these",
        "those", "it", "its", "they", "them", "their", "we", "our", "you",
        "your", "i", "my", "me", "he", "she", "his", "her", "not", "no",
        "so", "if", "then", "else", "when", "where", "what", "which", "who",
        "how", "all", "each", "every", "both", "few", "more", "most", "other",
        "some", "such", "only", "same", "than", "too", "very", "just", "also",
    }

    def __init__(
        self,
        variant: str = "okapi",
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
        use_stopwords: bool = True,
        custom_stopwords: Optional[set] = None,
        lowercase: bool = True,
        min_token_length: int = 2,
    ):
        """Initialize BM25 searcher.

        Args:
            variant: BM25 variant ('okapi', 'plus', 'l')
            k1: Term frequency saturation parameter
            b: Document length normalization parameter
            epsilon: Floor on IDF for BM25Plus
            use_stopwords: Whether to filter stopwords
            custom_stopwords: Additional stopwords to filter
            lowercase: Lowercase tokens
            min_token_length: Minimum token length to keep
        """
        if not BM25_AVAILABLE:
            raise ImportError("rank-bm25 required: pip install rank-bm25")

        self.variant = variant
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.use_stopwords = use_stopwords
        self.lowercase = lowercase
        self.min_token_length = min_token_length

        # Build stopword set
        self.stopwords = set()
        if use_stopwords:
            self.stopwords = self.DEFAULT_STOPWORDS.copy()
        if custom_stopwords:
            self.stopwords.update(custom_stopwords)

        # Will be set during indexing
        self.bm25 = None
        self.corpus_tokens = None
        self.chunk_ids = None

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25.

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        if self.lowercase:
            text = text.lower()

        # Split on non-alphanumeric (keep underscores for code)
        tokens = re.findall(r'\b[\w]+\b', text)

        # Filter
        tokens = [
            t for t in tokens
            if len(t) >= self.min_token_length
            and t not in self.stopwords
            and not t.isdigit()  # Remove pure numbers
        ]

        return tokens

    def index_chunks(
        self,
        chunks: List[Dict],
        content_fields: List[str] = None,
    ) -> None:
        """Build BM25 index from chunks.

        Args:
            chunks: List of chunk dicts (from chunker output)
            content_fields: Fields to include in tokenization.
                           Default: ["title", "content"]
        """
        if content_fields is None:
            content_fields = ["title", "content"]

        self.chunk_ids = []
        self.corpus_tokens = []

        for i, chunk in enumerate(chunks):
            # Build text from specified fields
            text_parts = []
            for field in content_fields:
                value = chunk.get(field, "")
                if isinstance(value, list):
                    value = " ".join(str(v) for v in value)
                text_parts.append(str(value))

            full_text = " ".join(text_parts)
            tokens = self.tokenize(full_text)

            self.corpus_tokens.append(tokens)
            self.chunk_ids.append(i)

        # Create BM25 index
        if self.variant == "plus":
            self.bm25 = BM25Plus(self.corpus_tokens, k1=self.k1, b=self.b)
        elif self.variant == "l":
            self.bm25 = BM25L(self.corpus_tokens, k1=self.k1, b=self.b)
        else:  # okapi (default)
            self.bm25 = BM25Okapi(self.corpus_tokens, k1=self.k1, b=self.b)

        print(f"BM25 index built: {len(self.corpus_tokens)} documents, "
              f"variant={self.variant}, k1={self.k1}, b={self.b}")

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Tuple[int, float]]:
        """Search for query.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of (chunk_index, score) tuples
        """
        if self.bm25 is None:
            raise ValueError("Index not built. Call index_chunks first.")

        query_tokens = self.tokenize(query)

        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = [
            (int(idx), float(scores[idx]))
            for idx in top_indices
            if scores[idx] > 0
        ]

        return results

    def batch_search(
        self,
        queries: Union[List[str], Dict[str, str]],
        top_k: int = 10,
    ) -> Union[List[List[Tuple[int, float]]], Dict[str, List[Tuple[int, float]]]]:
        """Search for multiple queries.

        Args:
            queries: List of query strings, or dict of {query_id: query_text}
            top_k: Number of results per query

        Returns:
            If list input: List of result lists
            If dict input: Dict of {query_id: results}
        """
        if isinstance(queries, dict):
            return {
                qid: self.search(query, top_k)
                for qid, query in queries.items()
            }
        else:
            return [self.search(query, top_k) for query in queries]

    def get_scores_array(
        self,
        queries: List[str],
    ) -> np.ndarray:
        """Get full score matrix for all queries.

        Args:
            queries: List of query strings

        Returns:
            numpy array of shape (len(queries), num_chunks)
        """
        if self.bm25 is None:
            raise ValueError("Index not built. Call index_chunks first.")

        all_scores = []
        for query in queries:
            query_tokens = self.tokenize(query)
            if query_tokens:
                scores = self.bm25.get_scores(query_tokens)
            else:
                scores = np.zeros(len(self.corpus_tokens))
            all_scores.append(scores)

        return np.array(all_scores)

    def get_document_frequency(self, term: str) -> int:
        """Get document frequency for a term.

        Args:
            term: Term to look up

        Returns:
            Number of documents containing the term
        """
        if self.bm25 is None:
            return 0

        term = term.lower() if self.lowercase else term
        return sum(1 for tokens in self.corpus_tokens if term in tokens)

    def get_corpus_stats(self) -> Dict:
        """Get corpus statistics.

        Returns:
            Dict with corpus stats
        """
        if self.bm25 is None:
            return {}

        doc_lengths = [len(tokens) for tokens in self.corpus_tokens]

        return {
            "num_documents": len(self.corpus_tokens),
            "avg_doc_length": np.mean(doc_lengths),
            "min_doc_length": min(doc_lengths),
            "max_doc_length": max(doc_lengths),
            "variant": self.variant,
            "k1": self.k1,
            "b": self.b,
        }


class BM25Ranker:
    """Wrapper for using BM25 in ranking strategies.

    Provides normalized scores and integration with fusion methods.
    """

    def __init__(
        self,
        searcher: Optional[BM25Searcher] = None,
        normalize: bool = True,
        **searcher_kwargs,
    ):
        """Initialize ranker.

        Args:
            searcher: Existing BM25Searcher, or None to create new
            normalize: Whether to normalize scores to [0, 1]
            **searcher_kwargs: Args for new BM25Searcher
        """
        self.searcher = searcher or BM25Searcher(**searcher_kwargs)
        self.normalize = normalize
        self._indexed = False

    def fit(self, chunks: List[Dict], **kwargs) -> "BM25Ranker":
        """Index chunks.

        Args:
            chunks: Chunks to index
            **kwargs: Additional args for index_chunks

        Returns:
            self for chaining
        """
        self.searcher.index_chunks(chunks, **kwargs)
        self._indexed = True
        return self

    def get_scores(
        self,
        query: str,
        normalize: Optional[bool] = None,
    ) -> np.ndarray:
        """Get BM25 scores for all chunks.

        Args:
            query: Query string
            normalize: Override default normalization

        Returns:
            Score array of shape (num_chunks,)
        """
        if not self._indexed:
            raise ValueError("Not indexed. Call fit() first.")

        query_tokens = self.searcher.tokenize(query)

        if not query_tokens:
            return np.zeros(len(self.searcher.corpus_tokens))

        scores = self.searcher.bm25.get_scores(query_tokens)

        should_normalize = normalize if normalize is not None else self.normalize
        if should_normalize and scores.max() > 0:
            scores = scores / scores.max()

        return scores

    def batch_get_scores(
        self,
        queries: List[str],
        normalize: Optional[bool] = None,
    ) -> np.ndarray:
        """Get BM25 scores for multiple queries.

        Args:
            queries: List of query strings
            normalize: Override default normalization

        Returns:
            Score matrix of shape (num_queries, num_chunks)
        """
        all_scores = self.searcher.get_scores_array(queries)

        should_normalize = normalize if normalize is not None else self.normalize
        if should_normalize:
            # Normalize each query's scores independently
            maxes = all_scores.max(axis=1, keepdims=True)
            maxes[maxes == 0] = 1  # Avoid division by zero
            all_scores = all_scores / maxes

        return all_scores


def create_bm25_searcher(
    variant: str = "okapi",
    **kwargs,
) -> BM25Searcher:
    """Factory function to create BM25 searcher.

    Args:
        variant: BM25 variant ('okapi', 'plus', 'l')
        **kwargs: Additional arguments

    Returns:
        BM25Searcher instance
    """
    return BM25Searcher(variant=variant, **kwargs)
