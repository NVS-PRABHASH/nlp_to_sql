"""
MRL RAG Implementation using txtai

This module provides RAG (Retrieval-Augmented Generation) functionality
for MRL (Management Reporting Language) queries using txtai for semantic search
and sentence-transformers/all-MiniLM-L6-v2 for embeddings.

Data Structure: Simple query-response pairs with columns ['query', 'hierarchy_data']
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
from config import get_config, setup_logging
setup_logging()
logger = logging.getLogger("ai_insight.mrl_rag")
logger.setLevel(logging.ERROR)


from txtai.embeddings import Embeddings
TXT_AI_AVAILABLE = True



class RAGSearch:
    """
    RAG (Retrieval-Augmented Generation) Search for query-hierarchy_data pairs using txtai.

    This class provides functionality to:
    1. Generate and cache document embeddings using sentence-transformers/all-MiniLM-L6-v2
    2. Search for relevant hierarchy_data based on natural language queries
    """

    def __init__(self, df: pd.DataFrame, cache_dir: str = '.rag_cache'):
        """
        Initialize the RAG search with a DataFrame of query-hierarchy_data pairs.

        Args:
            df: DataFrame containing query-hierarchy_data pairs with columns ['query', 'hierarchy_data']
            cache_dir: Directory to store cached embeddings
        """
        self.df = df.copy()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # Validate that we have the required columns
        required_columns = ['query', 'hierarchy_data']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(f"DataFrame must contain columns: {required_columns}. Missing: {missing_columns}")

        # Initialize txtai embeddings with sentence transformers
        # Use a basic configuration that should work with most txtai installations
        try:
            logger.info("Initializing txtai embeddings...")
            # Use the default configuration which should work with most installations
            self.embeddings = Embeddings(
                path="sentence-transformers/all-MiniLM-L6-v2",
                content=True
            )
            logger.info("Successfully initialized txtai embeddings")
        except Exception as e:
            logger.exception(f"Error initializing txtai embeddings: {e}")
            raise ValueError(f"Could not initialize txtai embeddings: {e}")

        # Prepare data for indexing
        self._prepare_data()

        # Build or load the index
        self._build_index()

    def _prepare_data(self):
        """Prepare the query-hierarchy_data pairs for indexing."""
        # Create unique IDs for each document
        self.df['_doc_id'] = range(len(self.df))

        # Prepare data for txtai indexing
        self.data = []
        for idx, row in self.df.iterrows():
            # Use the query as the searchable text
            query_text = str(row['query'])
            hierarchy_data = str(row['hierarchy_data'])

            if query_text.strip():
                self.data.append({
                    'id': str(row['_doc_id']),
                    'text': query_text,
                    'metadata': {
                        'query': query_text,
                        'hierarchy_data': hierarchy_data
                    }
                })

        logger.info(f"Prepared {len(self.data)} query-hierarchy_data pairs for indexing")

    def _build_index(self):
        """Build or load the txtai index."""
        index_file = self.cache_dir / 'txtai_index'

        if index_file.exists():
            try:
                logger.info("Loading existing txtai index...")
                self.embeddings.load(str(index_file))
                logger.info("Loaded txtai index from cache")
                return
            except Exception as e:
                logger.warning(f"Error loading index: {e}")

        # Build new index
        logger.info("Building txtai index with embeddings...")
        self.embeddings.index(self.data)
        logger.info("Built txtai index")

        # Save index
        try:
            self.embeddings.save(str(index_file))
            logger.info("Saved txtai index to cache")
        except Exception as e:
            logger.warning(f"Error saving index: {e}")

    def search(self, query: str, top_k: int = 10) -> pd.DataFrame:
        """
        Search for hierarchy_data relevant to the query using txtai.

        Args:
            query: Natural language query string
            top_k: Number of results to return

        Returns:
            DataFrame with top matching queries and their hierarchy_data
        """
        if not query or not isinstance(query, str) or not query.strip():
            return pd.DataFrame()

        try:
            # Perform semantic search using txtai
            results = self.embeddings.search(query, top_k)

            if not results:
                return pd.DataFrame()

            # Extract results
            matched_ids = [result['id'] for result in results]
            scores = [result['score'] for result in results]

            # Get the corresponding rows from the DataFrame
            results_df = self.df[self.df['_doc_id'].astype(str).isin(matched_ids)].copy()

            # Add similarity scores
            results_df['_similarity'] = 0.0
            for i, doc_id in enumerate(matched_ids):
                mask = results_df['_doc_id'].astype(str) == doc_id
                if mask.any():
                    results_df.loc[mask, '_similarity'] = scores[i]

            # Sort by similarity score
            results_df = results_df.sort_values('_similarity', ascending=False)

            return results_df

        except Exception as e:
            logger.exception(f"Error in txtai search: {e}")
            return pd.DataFrame()


class MRLChatbot:
    def __init__(self, data_file: str = None, cache_dir: str = None):
        """Initialize the MRL Chatbot with txtai RAG capabilities for query-hierarchy_data pairs.
        
        Args:
            data_file: Optional path to the MRL dataset CSV file. If not provided, will use the path from config.
            cache_dir: Optional directory for caching embeddings. If not provided, will use the path from config.
        """
        # Get configuration
        config = get_config()
        
        # Set paths from config if not provided
        self.data_file = data_file or config.get('mrl', {}).get('dataset_path', '../data/mrl_dataset.csv')
        self.cache_dir = cache_dir or config.get('mrl', {}).get('cache_dir', '.rag_cache')

        self.data = self._load_data()

        # Initialize RAG search with txtai
        if not self.data.empty:
            self.rag_search = RAGSearch(self.data, cache_dir=self.cache_dir)
        else:
            logger.warning("No data loaded for RAG")
            self.rag_search = None

    def _load_data(self) -> pd.DataFrame:
        """Load query-hierarchy_data pairs from file."""
        if not os.path.exists(self.data_file):
            logger.warning(f"Data file not found: {self.data_file}")
            return pd.DataFrame()

        try:
            if self.data_file.endswith('.json'):
                with open(self.data_file, 'r') as f:
                    return pd.DataFrame(json.load(f))
            elif self.data_file.endswith(('.xlsx', '.xls', '.csv')):
                return pd.read_csv(self.data_file) if self.data_file.endswith('.csv') else pd.read_excel(self.data_file)
            else:
                logger.warning(f"Unsupported file format: {self.data_file}")
                return pd.DataFrame()
        except Exception as e:
            logger.exception(f"Error loading data: {e}")
            return pd.DataFrame()

    def process_query(self, query: str) -> Dict:
        """
        Process a natural language query and return matching hierarchy_data.

        Args:
            query: Natural language query string

        Returns:
            Dictionary containing the results
        """
        logger.info(f"Processing query: {query}")

        if self.rag_search is None:
            return {'error': 'RAG search not initialized'}

        # Search using txtai RAG
        results = self.rag_search.search(query, top_k=10)

        if not results.empty:
            logger.info(f"Found {len(results)} potential matches using txtai RAG search")
            return {
                'hierarchy_data': results['hierarchy_data'].tolist(),
                'similarity_scores': results['_similarity'].tolist(),
                'source': 'txtai_rag_search',
                'details': results.to_dict('records')
            }

        # If no results from RAG, return error
        return {'error': 'No matching hierarchy_data found for the query'}

    def main(self):
        """Main entry point for the MRL Chatbot."""
        logger.info("MRL Chatbot with txtai RAG - Type 'exit' to quit")
        logger.info("Using txtai with sentence-transformers/all-MiniLM-L6-v2 embeddings for semantic search")
        logger.info(f"Loaded {len(self.data)} query-hierarchy_data pairs")

        while True:
            try:
                query = input("\n💬 Enter your query: ").strip()

                if query.lower() in ('exit', 'quit', 'q'):
                    break
                # Process the query
                result = self.process_query(query)
                # Display results
                if 'error' in result:
                    logger.error(result['error'])
                else:
                    logger.info(f"Found {len(result['hierarchy_data'])} matching results:")
                    
                    for i, (hierarchy, score, detail) in enumerate(zip(
                        result['hierarchy_data'],
                        result.get('similarity_scores', [0]*len(result['hierarchy_data'])),
                        result.get('details', [{}]*len(result['hierarchy_data']))
                    ), 1):
                        # Calculate the score percentage
                        score_pct = score * 100
                        
                        # Print the result
                        logger.info(f"Result {i} ({score_pct:.1f}% Match)")
                        logger.info("=" * 70)
                        logger.info(f"Query: {detail.get('query', 'N/A')}")
                        logger.info("Hierarchy Data: ")
                        
                        # Pretty print the JSON hierarchy
                        try:
                            import json
                            hierarchy_json = json.loads(hierarchy)
                            logger.info(" " + "\n ".join(json.dumps(hierarchy_json, indent=2).split('\n')))
                        except Exception:
                            logger.info(f" {hierarchy}")

            except KeyboardInterrupt:
                logger.info("Exiting...")
                break

