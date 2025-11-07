#!/usr/bin/env python3
"""
Domain Classifier for Banking Queries

This script classifies user queries as in-domain or out-of-domain using
TF-IDF vectorization and cosine similarity against known domain data.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import os
import sys
import json
import time


class DomainClassifier:
    """
    Lightweight domain classifier using TF-IDF and cosine similarity.
    
    Classifies queries as DOMAIN or OFF-DOMAIN based on similarity to
    known domain queries and banking-specific keywords.
    """
    
    def __init__(self, domain_data_path: str):
        """
        Initialize the classifier with domain data.
        
        Args:
            domain_data_path: Path to CSV file containing domain queries
        """
        self.domain_data_path = domain_data_path
        self.vectorizer = None
        self.domain_vectors = None
        self.domain_queries = []
        self.banking_keywords = set()
        
        # Load and prepare domain data
        self._load_domain_data()
        self._train_vectorizer()
    
    def _load_domain_data(self):
        """Load domain queries from CSV file."""
        try:
            df = pd.read_csv(self.domain_data_path)
            
            # Extract queries (first column)
            self.domain_queries = df['query'].dropna().tolist()
            
            # Extract banking-specific keywords from the queries
            self._extract_keywords()
            
            print(f"✓ Loaded {len(self.domain_queries)} domain queries")
            print(f"✓ Extracted {len(self.banking_keywords)} banking keywords")
            
        except Exception as e:
            print(f"✗ Error loading domain data: {e}")
            sys.exit(1)
    
    def _extract_keywords(self):
        """Extract banking-specific keywords from domain queries."""
        # Common banking/financial terms to boost
        base_keywords = {
            'assets', 'liabilities', 'equity', 'loan', 'deposit', 'account',
            'balance', 'bank', 'cash', 'credit', 'debit', 'interest', 'fee',
            'liability', 'provision', 'reserve', 'capital', 'investment',
            'revenue', 'expense', 'profit', 'loss', 'depreciation', 'amortization',
            'mortgage', 'overdraft', 'savings', 'checking', 'currency', 'foreign',
            'domestic', 'retail', 'corporate', 'wholesale', 'transaction',
            'customer', 'branch', 'atm', 'card', 'payment', 'transfer',
            'securities', 'bonds', 'treasury', 'guarantee', 'collateral', 'wise',
            'fixed', 'current', 'non-current', 'tangible', 'intangible','nfi',
            'nim','nii','pbt','pat','profit','loss','tax','net','margin','balance'
        }
        
        # Load keywords from keywords_and_rules.json
        try:
            keywords_file = os.path.join(os.path.dirname(self.domain_data_path), "..", "keywords_and_rules.json")
            keywords_file = os.path.normpath(keywords_file)
            
            if os.path.exists(keywords_file):
                with open(keywords_file, 'r', encoding='utf-8') as f:
                    keywords_data = json.load(f)
                    
                    # Extract keywords and aliases from keywords_and_rules
                    if "keywords_and_rules" in keywords_data:
                        for entry in keywords_data["keywords_and_rules"]:
                            # Add the main keyword
                            if "keyword" in entry:
                                keyword = entry["keyword"].lower()
                                # Extract words of 3+ characters (including alphanumeric for codes like 'q1', 'mtd')
                                words = re.findall(r'\b[a-z0-9]{2,}\b', keyword)
                                base_keywords.update(words)
                            
                            # Add all aliases
                            if "aliases" in entry and isinstance(entry["aliases"], list):
                                for alias in entry["aliases"]:
                                    alias_lower = str(alias).lower()
                                    # Extract words - include shorter terms and alphanumeric for technical terms
                                    words = re.findall(r'\b[a-z0-9_]{2,}\b', alias_lower)
                                    base_keywords.update(words)
                            
                            # Add tags as keywords too
                            if "tags" in entry and isinstance(entry["tags"], list):
                                for tag in entry["tags"]:
                                    tag_lower = str(tag).lower()
                                    # Extract words - include underscores for technical terms
                                    words = re.findall(r'\b[a-z0-9_]{2,}\b', tag_lower)
                                    base_keywords.update(words)
                    
                    # Extract quarter detection keywords
                    if "quarter_detection" in keywords_data:
                        quarter_data = keywords_data["quarter_detection"]
                        
                        # Add quarter aliases
                        if "quarter_aliases" in quarter_data:
                            for quarter_list in quarter_data["quarter_aliases"].values():
                                for phrase in quarter_list:
                                    words = re.findall(r'\b[a-z0-9]{2,}\b', phrase.lower())
                                    base_keywords.update(words)
                        
                        # Add previous year hints
                        if "py_hints" in quarter_data:
                            for hint in quarter_data["py_hints"]:
                                words = re.findall(r'\b[a-z0-9]{2,}\b', hint.lower())
                                base_keywords.update(words)
                
                print(f"✓ Loaded {len(base_keywords)} keywords from keywords_and_rules.json")
        except Exception as e:
            print(f"⚠ Warning: Could not load keywords_and_rules.json: {e}")
            print(f"  Continuing with base keywords only")
        
        # Extract unique words from domain queries
        for query in self.domain_queries:
            # Clean and tokenize
            words = re.findall(r'\b[a-z]{3,}\b', query.lower())
            self.banking_keywords.update(words)
        
        # Combine with base keywords
        self.banking_keywords.update(base_keywords)
    
    def _train_vectorizer(self):
        """Train TF-IDF vectorizer on domain queries."""
        # Use TF-IDF with word-level and character n-grams
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            analyzer='word',
            ngram_range=(1, 2),  # Unigrams and bigrams
            max_features=1000,   # Limit features for speed
            min_df=1,
            max_df=0.8
        )
        
        # Fit and transform domain queries
        self.domain_vectors = self.vectorizer.fit_transform(self.domain_queries)
        
        print(f"✓ Trained vectorizer with {self.domain_vectors.shape[1]} features")
    
    def _keyword_match_score(self, query: str) -> float:
        """
        Calculate keyword match score based on banking terms.
        
        Args:
            query: User query string
            
        Returns:
            Score between 0 and 1
        """
        query_words = set(re.findall(r'\b[a-z]{3,}\b', query.lower()))
        
        # Count matches
        matches = query_words.intersection(self.banking_keywords)
        
        if not query_words:
            return 0.0
        
        # Return ratio of matched keywords
        return len(matches) / len(query_words)
    
    def classify(self, query: str) -> tuple:
        """
        Classify a query as DOMAIN or OFF-DOMAIN.
        
        Args:
            query: User query string
            
        Returns:
            Tuple of (classification, confidence_percentage)
            - classification: "DOMAIN" or "OFF-DOMAIN"
            - confidence_percentage: float between 0 and 100
        """
        if not query or not query.strip():
            return "OFF-DOMAIN", 100.0
        
        # 1. Compute cosine similarity with domain queries
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.domain_vectors)
        
        # Get top similarity scores
        max_similarity = np.max(similarities)
        mean_top5_similarity = np.mean(np.sort(similarities[0])[-5:])
        
        # 2. Compute keyword match score
        keyword_score = self._keyword_match_score(query)
        
        # 3. Combine scores (weighted)
        # Cosine similarity: 60%, Keyword match: 40%
        combined_score = (0.6 * mean_top5_similarity) + (0.4 * keyword_score)
        
        # 4. Classification thresholds
        # High confidence: combined_score > 0.35
        # Medium confidence: 0.20 < combined_score <= 0.35
        # Low confidence (OFF-DOMAIN): combined_score <= 0.20
        
        if combined_score > 0.35:
            classification = "DOMAIN"
            confidence = min(50 + (combined_score * 100), 100)
        elif combined_score > 0.20:
            classification = "DOMAIN"
            confidence = 40 + (combined_score * 50)
        else:
            classification = "OFF-DOMAIN"
            confidence = 100 - (combined_score * 200)
        
        # Ensure confidence is in valid range
        confidence = max(0.0, min(100.0, confidence))
        
        return classification, round(confidence, 2)
    
    def get_classification_details(self, query: str) -> dict:
        """
        Get detailed classification information.
        
        Args:
            query: User query string
            
        Returns:
            Dictionary with classification details
        """
        classification, confidence = self.classify(query)
        
        # Get similarity scores
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.domain_vectors)[0]
        
        # Find top 3 similar queries
        top_indices = np.argsort(similarities)[-3:][::-1]
        top_similar = [
            {
                "query": self.domain_queries[idx],
                "similarity": round(similarities[idx] * 100, 2)
            }
            for idx in top_indices
        ]
        
        # Keyword analysis
        query_words = set(re.findall(r'\b[a-z]{3,}\b', query.lower()))
        matched_keywords = query_words.intersection(self.banking_keywords)
        keyword_score = self._keyword_match_score(query)
        
        return {
            "classification": classification,
            "confidence": confidence,
            "query": query,
            "details": {
                "keyword_match_score": round(keyword_score * 100, 2),
                "matched_keywords": list(matched_keywords)[:10],
                "top_similar_queries": top_similar
            }
        }


def main():
    """Main function to run interactive domain classifier."""
    
    # Determine the correct path to domain data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    domain_data_path = os.path.join(project_root, "data", "mrl_dataset.csv")
    
    # Check if file exists
    if not os.path.exists(domain_data_path):
        print(f"✗ Error: Domain data file not found at: {domain_data_path}")
        sys.exit(1)
    
    print("=" * 70)
    print("🏦 Banking Query Domain Classifier")
    print("=" * 70)
    print()
    
    # Initialize classifier
    print("Initializing classifier...")
    classifier = DomainClassifier(domain_data_path)
    print()
    
    print("=" * 70)
    print("Ready! Enter queries to classify (or 'quit' to exit)")
    print("=" * 70)
    print()
    
    # Interactive loop
    while True:
        try:
            # Get user input
            query = input("\n📝 Enter query: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            # Classify query
            start_time = time.perf_counter()
            classification, confidence = classifier.classify(query)
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            
            # Display result
            print()
            print("─" * 70)
            print(f"🔍 Query: {query}")
            print("─" * 70)
            
            # Color-coded output (using emojis for cross-platform support)
            if classification == "DOMAIN":
                icon = "✅"
            else:
                icon = "❌"
            
            print(f"{icon} Classification: {classification}")
            print(f"📊 Confidence: {confidence}%")
            print(f"⏱️ Elapsed Time: {elapsed_time*1000:.2f} ms")
            print()

            # Optional: Show detailed analysis
            # show_details = input("Show detailed analysis? (y/n): ").strip().lower()
            
            # if show_details == 'y':
            #     details = classifier.get_classification_details(query)
            #     print()
            #     print("📈 Detailed Analysis:")
            #     print(f"  • Keyword Match Score: {details['details']['keyword_match_score']}%")
            #     print(f"  • Matched Keywords: {', '.join(details['details']['matched_keywords'][:5])}")
            #     print()
            #     print("  Top Similar Domain Queries:")
            #     for i, similar in enumerate(details['details']['top_similar_queries'], 1):
            #         print(f"    {i}. [{similar['similarity']}%] {similar['query']}")
            #     print()
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n✗ Error: {e}")


if __name__ == "__main__":
    main()
