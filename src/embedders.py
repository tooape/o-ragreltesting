"""
EmbeddingGemma embedder with task-specific prompts for O-RAG.

Uses the prompt templates from EmbeddingGemma model card:
- Document: "title: {title} | text: {content}"
- Query (retrieval): "task: search result | query: {query}"
- Query (QA): "task: question answering | query: {query}"

Reference: https://ai.google.dev/gemma/docs/embeddinggemma/model_card
"""

from typing import List, Dict, Optional, Union
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False


class EmbeddingGemmaEmbedder:
    """EmbeddingGemma with O-RAG prompt templates.

    Supports Matryoshka dimension truncation (256, 512, or 768).
    Uses FP16 on GPU for faster inference.
    """

    MODEL_NAME = "google/embedding-gemma-001"
    NATIVE_DIM = 768

    # Prompt templates from model card
    DOC_TEMPLATE = "title: {title} | text: {text}"
    QUERY_TEMPLATE = "task: search result | query: {query}"
    QA_TEMPLATE = "task: question answering | query: {query}"

    def __init__(
        self,
        device: Optional[str] = None,
        truncate_dim: int = 256,
        use_fp16: bool = True,
    ):
        """Initialize EmbeddingGemma.

        Args:
            device: 'cuda', 'cuda:0', 'cpu', etc. Auto-detects if None.
            truncate_dim: Matryoshka dimension (256, 512, or 768)
            use_fp16: Use FP16 on GPU for faster inference
        """
        if not ST_AVAILABLE:
            raise ImportError("sentence-transformers required: pip install sentence-transformers")

        self.truncate_dim = truncate_dim
        self.use_fp16 = use_fp16

        # Auto-detect device
        if device is None:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device

        # Load model
        print(f"Loading {self.MODEL_NAME} on {self.device}...")
        self.model = SentenceTransformer(self.MODEL_NAME, device=self.device)

        # Enable FP16 for GPU
        if use_fp16 and "cuda" in self.device:
            self.model = self.model.half()
            print("Using FP16 for faster inference")

        print(f"Model loaded. Native dim: {self.NATIVE_DIM}, Truncated dim: {truncate_dim}")

    def encode_documents(
        self,
        chunks: List[Dict],
        batch_size: int = 512,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Encode document chunks with retrieval_document prompt.

        Args:
            chunks: List of {"title": str, "content": str} dicts
                   (from hierarchical chunker output)
            batch_size: Batch size for encoding
            show_progress: Show progress bar

        Returns:
            numpy array of shape (len(chunks), truncate_dim)
        """
        prompted = [
            self.DOC_TEMPLATE.format(
                title=c.get("title", "none"),
                text=c.get("content", c.get("text", ""))
            )
            for c in chunks
        ]
        return self._encode(prompted, batch_size, show_progress)

    def encode_queries(
        self,
        queries: Union[List[str], Dict[str, str]],
        task: str = "retrieval",
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Encode queries with appropriate task prompt.

        Args:
            queries: List of query strings, or dict of {query_id: query_text}
            task: 'retrieval' or 'qa'
            batch_size: Batch size for encoding
            show_progress: Show progress bar

        Returns:
            If list input: numpy array of shape (len(queries), truncate_dim)
            If dict input: dict of {query_id: embedding}
        """
        template = self.QA_TEMPLATE if task == "qa" else self.QUERY_TEMPLATE

        if isinstance(queries, dict):
            query_ids = list(queries.keys())
            query_texts = [queries[qid] for qid in query_ids]
            prompted = [template.format(query=q) for q in query_texts]
            embeddings = self._encode(prompted, batch_size, show_progress)
            return {qid: embeddings[i] for i, qid in enumerate(query_ids)}
        else:
            prompted = [template.format(query=q) for q in queries]
            return self._encode(prompted, batch_size, show_progress)

    def _encode(
        self,
        texts: List[str],
        batch_size: int,
        show_progress: bool,
    ) -> np.ndarray:
        """Internal encoding with Matryoshka truncation."""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        # Matryoshka truncation
        if self.truncate_dim and self.truncate_dim < self.NATIVE_DIM:
            embeddings = embeddings[:, :self.truncate_dim]
            # Renormalize after truncation
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-8)

        return embeddings

    @property
    def embedding_dim(self) -> int:
        """Get effective embedding dimension."""
        return self.truncate_dim if self.truncate_dim else self.NATIVE_DIM

    def save_embeddings(self, embeddings: np.ndarray, path: str):
        """Save embeddings to numpy file."""
        np.save(path, embeddings)
        print(f"Saved embeddings to {path}")

    def load_embeddings(self, path: str) -> np.ndarray:
        """Load embeddings from numpy file."""
        embeddings = np.load(path)
        print(f"Loaded embeddings from {path}: shape {embeddings.shape}")
        return embeddings


# Factory function for convenience
def create_embedder(
    model: str = "gemma",
    device: Optional[str] = None,
    truncate_dim: int = 256,
    **kwargs,
) -> EmbeddingGemmaEmbedder:
    """Factory function to create embedder.

    Args:
        model: Model type ('gemma' for EmbeddingGemma)
        device: Device to run on
        truncate_dim: Matryoshka dimension
        **kwargs: Additional arguments

    Returns:
        EmbeddingGemmaEmbedder instance
    """
    if model == "gemma":
        return EmbeddingGemmaEmbedder(
            device=device,
            truncate_dim=truncate_dim,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown model type: {model}")
