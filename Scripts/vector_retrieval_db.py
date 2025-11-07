"""
Unified Vector Retrieval Database

Centralized module for all vector-based retrieval operations:
- Semantic search for rules and keywords
- Example retrieval from prompt library
- Schema and column lookup
- MRL line retrieval
- Any other semantic similarity matching

Uses TF-IDF + Cosine Similarity for efficient vector search.
Can optionally use FAISS for large-scale similarity search.
"""

import json
import math
import re
import logging
import hashlib
import torch
from typing import List, Dict, Tuple, Optional, Set, Protocol
from dataclasses import dataclass, field
from pathlib import Path
from collections import Counter
from enum import Enum
from datetime import datetime

logger = logging.getLogger("ai_insight.vector_retrieval_db")
logger.setLevel(logging.ERROR)

# Try importing sklearn and FAISS for advanced features
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    SentenceTransformer = None  # type: ignore
    np = None  # type: ignore


# Cache embedding model so we only pay the load cost once per process
_EMBEDDING_MODEL: Optional[SentenceTransformer] = None  # type: ignore
_EMBEDDING_DEVICE: Optional[str] = None


def _get_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> Optional[SentenceTransformer]:  # type: ignore
    """Return a shared SentenceTransformer instance if available."""
    global _EMBEDDING_MODEL, _EMBEDDING_DEVICE
    if not HAS_SENTENCE_TRANSFORMERS:
        return None

    if _EMBEDDING_MODEL is None:
        try:
            chosen_model = model_name
            # Prefer GPU if available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _EMBEDDING_DEVICE = device
            # logger.info(f"Loading sentence-transformer model '{chosen_model}' on device '{device}' for semantic retrieval")
            _EMBEDDING_MODEL = SentenceTransformer(chosen_model, device=device)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning(f"Failed to load sentence-transformer model '{chosen_model}': {exc}")
            _EMBEDDING_MODEL = None
    return _EMBEDDING_MODEL


class VectorIndex(Protocol):
    """Protocol implemented by vector indices."""

    def search(self, query: str, k: int = 3, threshold: float = 0.0) -> List["VectorSearchResult"]:
        ...


class RetrievalType(Enum):
    """Types of retrieval operations supported."""
    RULES = "rules"
    EXAMPLES = "examples"
    KEYWORDS = "keywords"
    SCHEMA = "schema"
    MRL = "mrl"
    CUSTOM = "custom"


# ============================================================================
# Core Vector Index - TF-IDF Implementation
# ============================================================================

@dataclass
class VectorSearchResult:
    """Result from vector search operation."""
    id: str
    text: str
    score: float
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "text": self.text,
            "score": round(self.score, 4),
            "metadata": self.metadata
        }


class SemanticEmbeddingIndex:
    """Sentence-transformer powered semantic search index."""

    def __init__(self, texts: List[str], ids: Optional[List[str]] = None, metadata: Optional[List[Dict]] = None,
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.texts = texts
        self.ids = ids or [f"doc_{i}" for i in range(len(texts))]
        self.metadata = metadata or [{} for _ in texts]
        self.model_name = model_name
        self.model = _get_embedding_model(model_name)
        self.embeddings = None
        self.N = len(texts)

        if self.model is not None and texts:
            self._build_embeddings()
        else:
            logger.warning("Sentence-transformer model unavailable; embeddings index disabled")

    def _build_embeddings(self):
        try:
            # encode returns numpy array when convert_to_numpy=True
            self.embeddings = self.model.encode(
                self.texts,
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            # logger.info(f"Built semantic embedding index with {len(self.texts)} documents")
        except Exception as exc:
            logger.error(f"Failed to build embeddings index: {exc}")
            self.embeddings = None

    def search(self, query: str, k: int = 3, threshold: float = 0.0) -> List[VectorSearchResult]:
        if self.embeddings is None or self.model is None or np is None or not query:
            return []

        try:
            query_vec = self.model.encode(
                [query],
                batch_size=1,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )[0]
        except Exception as exc:
            logger.error(f"Failed to encode query for semantic search: {exc}")
            return []

        similarities = (self.embeddings @ query_vec).tolist()

        scored = [
            (idx, score)
            for idx, score in enumerate(similarities)
            if score >= threshold
        ]
        scored.sort(key=lambda item: item[1], reverse=True)

        results: List[VectorSearchResult] = []
        for idx, score in scored[:k]:
            results.append(
                VectorSearchResult(
                    id=self.ids[idx],
                    text=self.texts[idx],
                    score=float(score),
                    metadata=self.metadata[idx]
                )
            )
        return results

    def search_with_embedding(self, query_embedding, k: int = 3, threshold: float = 0.0) -> List[VectorSearchResult]:
        """Search using a precomputed query embedding to avoid duplicate encodes."""
        if self.embeddings is None or self.model is None or np is None or query_embedding is None:
            return []
        try:
            # Ensure correct shape
            if isinstance(query_embedding, list):
                query_vec = np.array(query_embedding, dtype=float)
            else:
                query_vec = query_embedding
            similarities = (self.embeddings @ query_vec).tolist()
        except Exception as exc:
            logger.error(f"Failed semantic search with precomputed embedding: {exc}")
            return []

        scored = [
            (idx, score)
            for idx, score in enumerate(similarities)
            if score >= threshold
        ]
        scored.sort(key=lambda item: item[1], reverse=True)

        results: List[VectorSearchResult] = []
        for idx, score in scored[:k]:
            results.append(
                VectorSearchResult(
                    id=self.ids[idx],
                    text=self.texts[idx],
                    score=float(score),
                    metadata=self.metadata[idx]
                )
            )
        return results


class SimpleTFIDFIndex:
    """
    Lightweight TF-IDF vector index with cosine similarity.
    No external dependencies required.
    """
    
    def __init__(self, texts: List[str], ids: Optional[List[str]] = None, 
                 metadata: Optional[List[Dict]] = None):
        """
        Initialize TFIDF index.
        
        Args:
            texts: List of text documents to index
            ids: Optional document IDs (default: doc_0, doc_1, ...)
            metadata: Optional metadata for each document
        """
        self.texts = texts
        self.ids = ids or [f"doc_{i}" for i in range(len(texts))]
        self.metadata = metadata or [{} for _ in texts]
        self.df = Counter()
        self.N = len(texts)
        self.vocabs: List[Dict[str, float]] = []
        
        if self.N > 0:
            self._build_index()
        
        # logger.info(f"Built TFIDF index with {self.N} documents")
    
    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Tokenize text to lowercase words."""
        return re.findall(r"[a-z0-9_]+", text.lower())
    
    def _build_index(self):
        """Build TF-IDF vectors for all documents."""
        # Calculate document frequencies
        docs_tokens = [self._tokenize(t) for t in self.texts]
        for tokens in docs_tokens:
            for term in set(tokens):
                self.df[term] += 1
        
        # Build vocabulary vectors with TF-IDF weights
        self.vocabs = []
        for tokens in docs_tokens:
            tf = Counter(tokens)
            vec = {}
            doc_len = len(tokens) or 1
            for term, freq in tf.items():
                # TF-IDF = (term_freq / doc_len) * IDF
                idf = math.log((1 + self.N) / (1 + self.df[term])) + 1.0
                vec[term] = (freq / doc_len) * idf
            self.vocabs.append(vec)
    
    @staticmethod
    def _cosine_similarity(vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = sum(vec_a.get(t, 0) * vec_b.get(t, 0) for t in set(vec_a) | set(vec_b))
        
        norm_a = math.sqrt(sum(v * v for v in vec_a.values())) or 1e-12
        norm_b = math.sqrt(sum(v * v for v in vec_b.values())) or 1e-12
        
        return dot_product / (norm_a * norm_b)
    
    def _embed_query(self, query: str) -> Dict[str, float]:
        """Convert query to TF-IDF vector."""
        tokens = self._tokenize(query)
        tf = Counter(tokens)
        vec = {}
        doc_len = len(tokens) or 1
        
        for term, freq in tf.items():
            idf = math.log((1 + self.N) / (1 + self.df.get(term, 0))) + 1.0
            vec[term] = (freq / doc_len) * idf
        
        return vec
    
    def search(self, query: str, k: int = 3, threshold: float = 0.0) -> List[VectorSearchResult]:
        """
        Search for most similar documents.
        
        Args:
            query: Search query text
            k: Number of results to return
            threshold: Minimum similarity score (0.0 to 1.0)
            
        Returns:
            List of VectorSearchResult sorted by similarity
        """
        if self.N == 0:
            return []
        
        query_vec = self._embed_query(query)
        scores = []
        
        for i, doc_vec in enumerate(self.vocabs):
            similarity = self._cosine_similarity(query_vec, doc_vec)
            if similarity >= threshold:
                scores.append((i, similarity))
        
        # Sort by similarity descending
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Build results
        results = []
        for idx, score in scores[:k]:
            result = VectorSearchResult(
                id=self.ids[idx],
                text=self.texts[idx],
                score=score,
                metadata=self.metadata[idx]
            )
            results.append(result)

        return results


# ============================================================================
# Unified Vector Retrieval Database
# ============================================================================

class VectorRetrievalDB:
    """
    Unified database for all vector-based retrieval operations.
    Manages multiple indices for different retrieval types.
    """
    
    def __init__(self):
        """Initialize the vector retrieval database."""
        self.indices: Dict[RetrievalType, VectorIndex] = {}
        self.config = {}
        self._semantic_enabled = HAS_SENTENCE_TRANSFORMERS and _get_embedding_model() is not None
        if not self._semantic_enabled:
            logger.info("VectorRetrievalDB falling back to TF-IDF retrieval")
    
    def add_index(self, retrieval_type: RetrievalType, texts: List[str], 
                  ids: Optional[List[str]] = None, metadata: Optional[List[Dict]] = None):
        """
        Add or update a vector index.
        
        Args:
            retrieval_type: Type of retrieval
            texts: List of documents to index
            ids: Optional document IDs
            metadata: Optional metadata per document
        """
        index: Optional[VectorIndex] = None

        if self._semantic_enabled:
            semantic_index = SemanticEmbeddingIndex(texts, ids, metadata)
            if getattr(semantic_index, "embeddings", None) is not None:
                index = semantic_index
            else:
                logger.warning(
                    "Semantic embeddings unavailable; reverting to TF-IDF for %s index",
                    retrieval_type.value
                )

        if index is None:
            index = SimpleTFIDFIndex(texts, ids, metadata)

        self.indices[retrieval_type] = index
        engine = "sentence_transformer" if isinstance(index, SemanticEmbeddingIndex) else "tfidf"
        # logger.info(
        #     "Added %s index with %d documents using %s engine",
        #     retrieval_type.value,
        #     len(texts),
        #     engine
        # )
    
    def encode_query(self, query: str):
        """Encode a query once using the global embedding model (if available)."""
        model = _get_embedding_model()
        if model is None or not query:
            return None
        try:
            vec = model.encode([query], batch_size=1, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)[0]
            return vec
        except Exception as exc:
            logger.error(f"Failed to encode query: {exc}")
            return None
    
    def search(self, retrieval_type: RetrievalType, query: str, k: int = 3,
               threshold: float = 0.0) -> List[VectorSearchResult]:
        """
        Search using a specific index.
        
        Args:
            retrieval_type: Which index to search
            query: Search query
            k: Number of results
            threshold: Minimum similarity score
            
        Returns:
            List of search results
        """
        if retrieval_type not in self.indices:
            logger.warning(f"No index found for {retrieval_type.value}")
            return []
        
        return self.indices[retrieval_type].search(query, k=k, threshold=threshold)

    def search_with_embedding(self, retrieval_type: RetrievalType, query_embedding, k: int = 3,
                              threshold: float = 0.0) -> List[VectorSearchResult]:
        """Search using a precomputed embedding when the index supports it."""
        if retrieval_type not in self.indices:
            logger.warning(f"No index found for {retrieval_type.value}")
            return []
        index = self.indices[retrieval_type]
        if isinstance(index, SemanticEmbeddingIndex):
            return index.search_with_embedding(query_embedding, k=k, threshold=threshold)
        # Fallback: no embedding path for TF-IDF; this will re-embed internally
        return index.search("", k=k, threshold=threshold)
    
    def search_all(self, query: str, k: int = 3) -> Dict[str, List[VectorSearchResult]]:
        """
        Search across all indices.
        
        Args:
            query: Search query
            k: Results per index
            
        Returns:
            Dict mapping retrieval type to results
        """
        results = {}
        for retrieval_type in self.indices:
            results[retrieval_type.value] = self.search(retrieval_type, query, k=k)
        return results
    
    def get_stats(self) -> Dict:
        """Get statistics about the database."""
        stats = {
            "total_indices": len(self.indices),
            "indices": {}
        }
        for ret_type, index in self.indices.items():
            if isinstance(index, SemanticEmbeddingIndex):
                stats["indices"][ret_type.value] = {
                    "num_documents": index.N,
                    "engine": "sentence_transformer",
                    "model": index.model_name,
                }
            else:
                stats["indices"][ret_type.value] = {
                    "num_documents": index.N,
                    "engine": "tfidf"
                }
        return stats


# ============================================================================
# Specialized Retrieval Interfaces
# ============================================================================

class RuleRetriever:
    """Vector-based rule retrieval for prompt generation."""
    
    def __init__(self, vdb: VectorRetrievalDB):
        self.vdb = vdb
    
    def build_from_json(self, rules_path: Path):
        """Build rule index from JSON configuration."""
        try:
            with open(rules_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            texts = []
            ids = []
            metadata = []
            
            for item in config.get('keywords_and_rules', []):
                keyword = item.get('keyword', '')
                rule_text = item.get('rule', '')
                section = item.get('section', 'general')
                priority = item.get('priority', 5)
                
                if keyword and rule_text:
                    texts.append(rule_text)
                    ids.append(keyword)
                    metadata.append({
                        'section': section,
                        'priority': priority,
                        'original_keyword': keyword
                    })
            
            self.vdb.add_index(RetrievalType.RULES, texts, ids, metadata)
            # logger.info(f"Built rule index with {len(texts)} rules")
        
        except Exception as e:
            logger.error(f"Failed to build rule index: {e}")
    
    def retrieve_rules(self, query: str, k: int = 5) -> List[VectorSearchResult]:
        """Retrieve relevant rules using semantic similarity."""
        return self.vdb.search(RetrievalType.RULES, query, k=k, threshold=0.1)

    def retrieve_rules_with_embedding(self, query_embedding, k: int = 5) -> List[VectorSearchResult]:
        """Retrieve relevant rules using a precomputed embedding."""
        return self.vdb.search_with_embedding(RetrievalType.RULES, query_embedding, k=k, threshold=0.1)


class ExampleRetriever:
    """Vector-based example retrieval from prompt library."""
    
    def __init__(self, vdb: VectorRetrievalDB):
        self.vdb = vdb
        self.examples_path = None
        self.prompt_lib_hash = None
        self.last_rebuild_time = None
        self.example_count = 0
    
    def _compute_file_hash(self, file_path: Path) -> Optional[str]:
        """
        Compute SHA256 hash of prompt_lib.json file.
        Used to detect when file changes and index needs rebuild.
        
        Args:
            file_path: Path to prompt_lib.json
            
        Returns:
            SHA256 hash string or None if file not found
        """
        try:
            with open(file_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except FileNotFoundError:
            logger.warning(f"File not found for hashing: {file_path}")
            return None
        except Exception as e:
            logger.error(f"Error computing file hash: {e}")
            return None
    
    def rebuild_index(self, examples_path: Path):
        """
        Manually rebuild the examples index from prompt_lib.json.
        Call this when you've added new examples to prompt_lib.json.
        
        Args:
            examples_path: Path to prompt_lib.json
        """
        # logger.info("🔄 Rebuilding examples index from prompt_lib.json...")
        
        self.examples_path = examples_path
        self.build_from_json(examples_path)
        
        # Update hash and timestamp after build
        self.prompt_lib_hash = self._compute_file_hash(examples_path)
        self.last_rebuild_time = datetime.now()
        
        # logger.info(f"✅ Examples index rebuilt at {self.last_rebuild_time.isoformat()}")
    
    def _check_and_rebuild_if_needed(self) -> bool:
        """
        Check if prompt_lib.json has changed since last build.
        If changed, rebuild index automatically.
        
        Returns:
            True if rebuild was performed, False otherwise
        """
        if not self.examples_path:
            return False
        
        current_hash = self._compute_file_hash(self.examples_path)
        
        if current_hash is None:
            return False
        
        if current_hash != self.prompt_lib_hash:
            # logger.info(
            #     f"📝 Detected changes in {self.examples_path.name}\n"
            #     f"   Previous hash: {self.prompt_lib_hash[:8]}...\n"
            #     f"   Current hash:  {current_hash[:8]}...\n"
            #     f"   Rebuilding index..."
            # )
            self.rebuild_index(self.examples_path)
            return True
        
        return False
    
    def build_from_json(self, examples_path: Path):
        """Build examples index from JSON prompt library."""
        try:
            self.examples_path = examples_path
            with open(examples_path, 'r', encoding='utf-8') as f:
                library = json.load(f)
            
            texts = []
            ids = []
            metadata = []
            full_texts = []  # Store full text for retrieval
            
            for snippet in library.get('snippets', []):
                snippet_id = snippet.get('id', '')
                full_text = snippet.get('text', '')
                tags = snippet.get('tags', [])
                
                if snippet_id and full_text:
                    # Extract query part only (before SQL:) for better TF-IDF matching
                    # This focuses similarity matching on what the user asked, not SQL syntax
                    if 'SQL' in full_text:
                        query_text = full_text.split('SQL')[0].replace('User:', '').replace('User ', '').strip()
                    else:
                        query_text = full_text
                    
                    # Use query text for indexing (better relevance)
                    texts.append(query_text)
                    ids.append(snippet_id)
                    metadata.append({'tags': tags, 'full_text': full_text})
                    full_texts.append(full_text)
            
            self.vdb.add_index(RetrievalType.EXAMPLES, texts, ids, metadata)
            self.example_count = len(texts)
            
            # Store initial hash if not already set
            if not self.prompt_lib_hash:
                self.prompt_lib_hash = self._compute_file_hash(examples_path)
            
            if not self.last_rebuild_time:
                self.last_rebuild_time = datetime.now()
            
            # logger.info(f"Built examples index with {len(texts)} examples")
        
        except Exception as e:
            logger.error(f"Failed to build examples index: {e}")
    
    def retrieve_examples(self, query: str, k: int = 2, tags: Optional[List[str]] = None) -> List[VectorSearchResult]:
        """
        Retrieve relevant examples using semantic similarity.
        Automatically detects and rebuilds index if prompt_lib.json has changed.
        
        Args:
            query: Search query
            k: Number of examples to retrieve
            tags: Optional tags to filter by
            
        Returns:
            List of VectorSearchResult with top-k similar examples
        """
        # Check if index needs rebuild (automatic detection)
        rebuilt = self._check_and_rebuild_if_needed()
        
        if rebuilt:
            logger.info(f"✅ Index updated. Retrieving {k} examples...")

        results = self.vdb.search(RetrievalType.EXAMPLES, query, k=k*2, threshold=0.0)
        
        # Restore full text from metadata for display
        for result in results:
            if 'full_text' in result.metadata:
                result.text = result.metadata['full_text']
        
        # Filter by tags if specified
        if tags:
            filtered = []
            for result in results:
                result_tags = result.metadata.get('tags', [])
                if any(t in result_tags for t in tags):
                    filtered.append(result)
                if len(filtered) >= k:
                    break
            return filtered
        
        return results[:k]

    def retrieve_examples_with_embedding(self, query_embedding, k: int = 2, tags: Optional[List[str]] = None) -> List[VectorSearchResult]:
        """Retrieve examples using a precomputed embedding; preserves tag filtering."""
        rebuilt = self._check_and_rebuild_if_needed()
        if rebuilt:
            logger.info(f"✅ Index updated. Retrieving {k} examples...")

        results = self.vdb.search_with_embedding(RetrievalType.EXAMPLES, query_embedding, k=k*2, threshold=0.0)

        # Restore full text from metadata for display
        for result in results:
            if 'full_text' in result.metadata:
                result.text = result.metadata['full_text']

        if tags:
            filtered = []
            for result in results:
                result_tags = result.metadata.get('tags', [])
                if any(t in result_tags for t in tags):
                    filtered.append(result)
                if len(filtered) >= k:
                    break
            return filtered
        return results[:k]
    
    def get_status(self) -> Dict:
        """
        Get current status of the example retriever.
        Shows example count, last rebuild time, and whether index is current.
        Useful for monitoring and debugging.
        
        Returns:
            Dict with status information:
                - examples_count: Number of examples in index
                - last_rebuild: ISO format timestamp of last rebuild
                - prompt_lib_hash: First 8 chars of current file hash
                - is_current: Whether index matches current prompt_lib.json
                - examples_path: Path to prompt_lib.json
        """
        is_current = True
        if self.examples_path and self.prompt_lib_hash:
            current_hash = self._compute_file_hash(self.examples_path)
            is_current = (current_hash == self.prompt_lib_hash)
        
        return {
            'examples_count': self.example_count,
            'last_rebuild': self.last_rebuild_time.isoformat() if self.last_rebuild_time else None,
            'prompt_lib_hash': self.prompt_lib_hash[:8] + "..." if self.prompt_lib_hash else None,
            'is_current': is_current,
            'examples_path': str(self.examples_path) if self.examples_path else None
        }


class SchemaRetriever:
    """Vector-based schema and column lookup."""
    
    def __init__(self, vdb: VectorRetrievalDB):
        self.vdb = vdb
    
    def build_from_json(self, schema_path: Path):
        """Build schema index from allowed_schema.json."""
        try:
            with open(schema_path, 'r', encoding='utf-8') as f:
                schema = json.load(f)
            
            texts = []
            ids = []
            metadata = []
            
            for table, columns in schema.items():
                for col in columns:
                    # Create comprehensive text representations
                    col_text = f"{table} {col} {col.lower()} {col.upper()}"
                    texts.append(col_text)
                    ids.append(f"{table}.{col}")
                    metadata.append({
                        'table': table,
                        'column': col,
                        'type': 'column'
                    })
            
            self.vdb.add_index(RetrievalType.SCHEMA, texts, ids, metadata)
            # logger.info(f"Built schema index with {len(texts)} columns")
        
        except Exception as e:
            logger.error(f"Failed to build schema index: {e}")
    
    def retrieve_columns(self, query: str, k: int = 5) -> List[VectorSearchResult]:
        """Find relevant columns for a query."""
        return self.vdb.search(RetrievalType.SCHEMA, query, k=k, threshold=0.2)

    def retrieve_columns_with_embedding(self, query_embedding, k: int = 5) -> List[VectorSearchResult]:
        """Find relevant columns using a precomputed embedding."""
        return self.vdb.search_with_embedding(RetrievalType.SCHEMA, query_embedding, k=k, threshold=0.2)


class MRLRetriever:
    """Vector-based MRL (Measure/Risk Line) retrieval."""
    
    def __init__(self, vdb: VectorRetrievalDB):
        self.vdb = vdb
        self.mrl_descriptions = {}
    
    def build_from_dict(self, mrl_data: Dict[str, str]):
        """
        Build MRL index from dictionary of MRL codes to descriptions.
        
        Args:
            mrl_data: Dict mapping MRL code to description
        """
        texts = []
        ids = []
        metadata = []
        
        for mrl_code, description in mrl_data.items():
            # Create rich text representation
            mrl_text = f"{mrl_code} {description}"
            texts.append(mrl_text)
            ids.append(mrl_code)
            metadata.append({
                'mrl_code': mrl_code,
                'description': description,
                'type': 'mrl'
            })
            self.mrl_descriptions[mrl_code] = description
        
        self.vdb.add_index(RetrievalType.MRL, texts, ids, metadata)
        # logger.info(f"Built MRL index with {len(texts)} MRL codes")
    
    def retrieve_mrl(self, query: str, k: int = 5) -> List[VectorSearchResult]:
        """Find relevant MRL codes for a query."""
        return self.vdb.search(RetrievalType.MRL, query, k=k, threshold=0.15)


class KeywordRetriever:
    """Vector-based keyword retrieval and expansion."""
    
    def __init__(self, vdb: VectorRetrievalDB):
        self.vdb = vdb
    
    def build_from_json(self, keywords_path: Path):
        """Build keyword index from keywords_and_rules.json."""
        try:
            with open(keywords_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            texts = []
            ids = []
            metadata = []
            
            for item in config.get('keywords_and_rules', []):
                keyword = item.get('keyword', '')
                aliases = item.get('aliases', [])
                
                if keyword:
                    # Create text combining keyword and all aliases
                    keyword_text = f"{keyword} {' '.join(aliases)}"
                    texts.append(keyword_text)
                    ids.append(keyword)
                    metadata.append({
                        'keyword': keyword,
                        'aliases': aliases,
                        'num_aliases': len(aliases)
                    })
            
            self.vdb.add_index(RetrievalType.KEYWORDS, texts, ids, metadata)
            # logger.info(f"Built keyword index with {len(texts)} keywords")
        
        except Exception as e:
            logger.error(f"Failed to build keyword index: {e}")
    
    def expand_keywords(self, query: str, k: int = 5) -> List[VectorSearchResult]:
        """Expand query keywords using semantic similarity."""
        # Lower threshold for small corpus (was 0.2, now 0.50)
        return self.vdb.search(RetrievalType.KEYWORDS, query, k=k, threshold=0.50)

    def expand_keywords_with_embedding(self, query_embedding, k: int = 5) -> List[VectorSearchResult]:
        """Expand keywords using precomputed embedding."""
        return self.vdb.search_with_embedding(RetrievalType.KEYWORDS, query_embedding, k=k, threshold=0.50)


# ============================================================================
# Singleton Instance
# ============================================================================

_global_vdb: Optional[VectorRetrievalDB] = None
_global_retrievers: Dict[str, object] = {}


def get_vector_db() -> VectorRetrievalDB:
    """Get global vector database instance."""
    global _global_vdb
    if _global_vdb is None:
        _global_vdb = VectorRetrievalDB()
    return _global_vdb


def initialize_all_retrievers(workspace_root: Path) -> Dict[str, object]:
    """
    Initialize all retrievers with default paths.
    
    Args:
        workspace_root: Root directory of workspace
        
    Returns:
        Dict of retriever instances
    """
    global _global_retrievers
    
    if _global_retrievers:
        return _global_retrievers
    
    vdb = get_vector_db()
    
    # Initialize all retrievers
    _global_retrievers['rules'] = RuleRetriever(vdb)
    _global_retrievers['examples'] = ExampleRetriever(vdb)
    _global_retrievers['schema'] = SchemaRetriever(vdb)
    _global_retrievers['keywords'] = KeywordRetriever(vdb)
    _global_retrievers['mrl'] = MRLRetriever(vdb)
    
    # Build indices from files
    try:
        _global_retrievers['rules'].build_from_json(workspace_root / "keywords_and_rules.json")
        _global_retrievers['examples'].build_from_json(workspace_root / "prompt_lib.json")
        _global_retrievers['schema'].build_from_json(workspace_root / "allowed_schema.json")
        _global_retrievers['keywords'].build_from_json(workspace_root / "keywords_and_rules.json")
        
        # logger.info("All retrievers initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize retrievers: {e}")
    
    return _global_retrievers


# ============================================================================
# Testing & Demonstration
# ============================================================================

# if __name__ == "__main__":
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
#     )
    
#     print("\n" + "="*80)
#     print("VECTOR RETRIEVAL DATABASE - DEMONSTRATION")
#     print("="*80)
    
#     workspace_root = Path(__file__).parent.parent
#     retrievers = initialize_all_retrievers(workspace_root)
    
#     vdb = get_vector_db()
#     print(f"\n📊 Database Statistics:")
#     print(json.dumps(vdb.get_stats(), indent=2))
    
#     # Test queries
#     test_queries = {
#         "rules": "deposits filtering criteria",
#         "examples": "term loan calculation",
#         "schema": "business day column",
#         "keywords": "year to date comparison",
#         "mrl": "term loan products"
#     }
    
#     print(f"\n{'='*80}")
#     print("VECTOR SEARCH TESTS")
#     print(f"{'='*80}")
    
#     for ret_type, query in test_queries.items():
#         print(f"\n🔍 {ret_type.upper()}: {query}")
#         if ret_type == "rules":
#             results = retrievers['rules'].retrieve_rules(query, k=3)
#         elif ret_type == "examples":
#             results = retrievers['examples'].retrieve_examples(query, k=2)
#         elif ret_type == "schema":
#             results = retrievers['schema'].retrieve_columns(query, k=3)
#         elif ret_type == "keywords":
#             results = retrievers['keywords'].expand_keywords(query, k=3)
#         else:
#             results = vdb.search(RetrievalType.MRL, query, k=3)
        
#         for i, result in enumerate(results, 1):
#             print(f"  {i}. [{result.score:.3f}] {result.id}")
#             print(f"     {result.text[:80]}...")
    
#     print("\n" + "="*80)
#     print("✅ Vector Retrieval System Ready")
#     print("="*80 + "\n")
