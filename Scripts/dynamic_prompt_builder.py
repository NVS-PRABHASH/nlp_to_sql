"""
Dynamic Prompt Builder with Vector Retrieval

This module implements dynamic prompt construction using vector-based semantic search.
Uses unified vector retrieval system for all keyword and rule matching.

Architecture:
- Extract query keywords using vector semantic search
- Retrieve relevant rules using vector similarity
- Retrieve relevant examples using vector similarity
- Compose prompt with only relevant sections and rules
- Reduce token count and improve prompt specificity

All retrieval now uses TF-IDF cosine similarity from vector_retrieval_db.py
"""

import json
import re
import logging
from typing import List, Dict, Set, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
import os
import sys

# Ensure Scripts directory is in path for imports
_scripts_dir = Path(__file__).parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

# Import unified vector retrieval system
from vector_retrieval_db import (
    get_vector_db, 
    initialize_all_retrievers,
    RuleRetriever,
    ExampleRetriever,
    KeywordRetriever,
    VectorSearchResult
)

# Import quarter rules handler from time_reference module
from time_reference import inject_quarter_rules_into_prompt
from month_mapper import inject_month_rules_into_prompt

logger = logging.getLogger("ai_insight.dynamic_prompt_builder")
logger.setLevel(logging.ERROR)

# ============================================================================
# Configuration File Paths
# ============================================================================

WORKSPACE_ROOT = Path(__file__).parent.parent
PROMPT_TEMPLATE_PATH = WORKSPACE_ROOT / "prompt_template.json"
RULES_KEYWORDS_PATH = WORKSPACE_ROOT / "keywords_and_rules.json"

# Initialize vector retrieval system
_vdb = None
_retrievers = None

def _initialize_vector_system():
    """Initialize the vector retrieval system on first use."""
    global _vdb, _retrievers
    if _vdb is None:
        try:
            _vdb = get_vector_db()
            _retrievers = initialize_all_retrievers(WORKSPACE_ROOT)
            logger.info("Vector retrieval system initialized")
        except Exception as e:
            logger.error(f"Failed to initialize vector system: {e}")

# ============================================================================
# Data Structures for Dynamic Prompt Rules
# ============================================================================

@dataclass
class PromptRule:
    """Represents a single prompt rule for a specific keyword/concept."""
    keyword: str
    rule_text: str
    section: str  # e.g., "selection", "aggregation", "filter", "time_handling"
    priority: int = 0  # Higher priority rules override lower ones
    tags: List[str] = field(default_factory=list)
    score: float = 0.0  # Vector similarity score
    
    def to_dict(self) -> Dict:
        return {
            "keyword": self.keyword,
            "rule_text": self.rule_text,
            "section": self.section,
            "priority": self.priority,
            "tags": self.tags,
            "score": round(self.score, 4)
        }


@dataclass
class PromptSection:
    """Represents a section of the final prompt."""
    name: str
    description: str
    rules: List[PromptRule] = field(default_factory=list)
    template: str = ""  # Template text with placeholders
    
    def render(self) -> str:
        """Render section with its title, description, and rules."""
        # Build section header with title in uppercase
        section_title = self.name.upper().replace("_", " ")
        section_output = f"\n==== {section_title} ===="
        
        # Add description if provided
        if self.description:
            section_output += f"\n{self.description}"
        
        # Add rules if present
        if self.rules:
            rules_text = "\n".join([f"- {rule.rule_text}" for rule in self.rules])
            section_output += f"\n{rules_text}"
        
        return section_output


@dataclass
class DynamicPrompt:
    """Complete dynamic prompt with all sections."""
    system_prompt: str
    footer_prompt: str
    sections: Dict[str, PromptSection]
    user_query: str
    keywords: List[str]
    metadata: Dict = field(default_factory=dict)
    section_orders: Dict[str, int] = field(default_factory=dict)  # Stores section order values
    
    def render(self) -> str:
        """
        Render complete prompt with all sections ordered by their 'order' field.
        Preserves the structure defined in prompt_template.json.
        """
        # Render all sections that have rules (important) or descriptions
        sections_with_content = []
        
        for section_name, section in self.sections.items():
            # Include section if it has rules OR has important description
            if section.rules or (section.description and len(section.description) > 10):
                rendered = section.render()
                if rendered:
                    # Store tuple of (order, section_name, rendered_text)
                    # Default to 999 for sections without order to place them at the end
                    order = self.section_orders.get(section_name, 999)
                    sections_with_content.append((order, section_name, rendered))
        
        # Sort sections by their order value
        sections_with_content.sort(key=lambda x: x[0])
        
        # Extract just the rendered text in the sorted order
        sections_text = "\n".join([section[2] for section in sections_with_content])
        
        return f"""{self.system_prompt}

{sections_text}

{self.footer_prompt}
"""


# ============================================================================
# Vector-Based Keyword Extraction Engine
# ============================================================================

class KeywordExtractor:
    """
    Extract keywords from user queries using vector semantic similarity.
    Uses unified vector retrieval system for semantic keyword matching.
    """
    
    def __init__(self):
        """Initialize with vector retrieval system."""
        _initialize_vector_system()
        self.keyword_retriever: Optional[KeywordRetriever] = _retrievers.get('keywords') if _retrievers else None
    
    def extract(self, query: str, k: int = 10, threshold: float = 0.15) -> List[str]:
        """
        Extract keywords from query using vector semantic search.
        
        Args:
            query: User query text
            k: Number of keyword candidates to retrieve
            threshold: Minimum similarity threshold
            
        Returns:
            List of canonical keywords extracted from query
        """
        if not self.keyword_retriever:
            logger.warning("Keyword retriever not available, using fallback extraction")
            return self._fallback_extract(query)

    def extract_with_embedding(self, query_embedding, k: int = 10, threshold: float = 0.15) -> List[str]:
        """Extract keywords using a precomputed query embedding to avoid duplicate encodes."""
        keywords = []
        seen = set()
        
        # First, try vector-based extraction if available
        if self.keyword_retriever and query_embedding is not None:
            try:
                results = self.keyword_retriever.expand_keywords_with_embedding(query_embedding, k=k)
                if results:  # Check if results is not None
                    for result in results:
                        keyword = result.id
                        if keyword not in seen and result.score >= threshold:
                            keywords.append(keyword)
                            seen.add(keyword)
                    # logger.info(f"Extracted {len(keywords)} keywords using precomputed embedding")
            except Exception as e:
                logger.error(f"Vector keyword extraction with embedding failed: {e}")
        
        # Return whatever we got from vector search (may be empty)
        return keywords
        
        try:
            # Use vector search to find similar keywords
            results = self.keyword_retriever.expand_keywords(query, k=k)
            
            # Extract canonical keywords from results
            keywords = []
            seen = set()
            for result in results:
                keyword = result.id  # ID is the canonical keyword
                if keyword not in seen and result.score >= threshold:
                    keywords.append(keyword)
                    seen.add(keyword)
                    logger.debug(f"Keyword: {keyword} (score: {result.score:.3f})")
            
            logger.info(f"Extracted {len(keywords)} keywords using vector search")
            return keywords
        
        except Exception as e:
            logger.error(f"Vector keyword extraction failed: {e}, using fallback")
            return self._fallback_extract(query)
    
    def _fallback_extract(self, query: str) -> List[str]:
        """
        Fallback simple extraction if vector system unavailable.
        Used for backward compatibility only.
        """
        keywords = []
        stop_words = {"what", "is", "the", "a", "an", "and", "or", "in", "of", "to", "as", "for", "by"}
        
        # Simple word extraction
        words = re.findall(r'\b\w+\b', query.lower())
        for word in words:
            if word not in stop_words and len(word) > 2:
                keywords.append(word)
        
        return list(dict.fromkeys(keywords))  # Remove duplicates while preserving order


# ============================================================================
# Vector-Based Prompt Rules Database
# ============================================================================

class PromptRulesDB:
    """
    Vector database of prompt rules using unified retrieval system.
    All rule retrieval now uses semantic vector similarity.
    
    Enhanced to support keyword-to-rule mapping through aliases and section-based organization.
    """
    
    def __init__(self):
        """Initialize with vector retrieval system."""
        _initialize_vector_system()
        self.rule_retriever: Optional[RuleRetriever] = _retrievers.get('rules') if _retrievers else None
        self.keyword_to_rules: Dict[str, List[PromptRule]] = {}  # keyword -> rules
        self.alias_to_keywords: Dict[str, List[str]] = {}  # alias -> list of canonical keywords
        self.section_rules: Dict[str, List[PromptRule]] = {}  # section -> all rules
        # Selection tuning to reduce unnecessary rules in prompts
        # Caps are soft limits applied per section after sorting by priority and score
        self.selection_caps: Dict[str, int] = {
            "core_dimensions":4,
            "account_dimensions":11,
            "period_mapping":10,
            "financial_measures":10,
            "additional_filter_rules":10,
            "bal_type_selection_rules":10,
            "comparison_growth_logic":10,
            "mrl_exclusion_rule":10,
            "trend_rules":10,
            "safe_division_rule":10,
            "yoy_growth_calculation":10,
            "multi_month_queries":10,
            "important_clarification":10,
            "case_multiple_products":10,
            "case_multiple_sbus":10,
            "customer_dimensions":10
        }
        # Default stricter minimum similarity for keyword->rule retrieval
        self.default_min_similarity: float = 0.25
        self._load_rules_from_json()  # Load all rules from keywords_and_rules.json
    
    def _load_rules_from_json(self):
        """Load all rules and keyword mappings from keywords_and_rules.json."""
        try:
            if not RULES_KEYWORDS_PATH.exists():
                logger.warning(f"Rules keywords file not found: {RULES_KEYWORDS_PATH}")
                return
            
            with open(RULES_KEYWORDS_PATH, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # logger.info(f"📚 Loading rules from keywords_and_rules.json...")
            
            for item in config.get('keywords_and_rules', []):
                keyword = item.get('keyword', '')
                section = item.get('section', 'general')
                priority = item.get('priority', 5)
                tags = item.get('tags', [])
                rule_text = item.get('rule', '')
                aliases = item.get('aliases', [])
                
                if keyword and rule_text:
                    # Create rule object
                    rule = PromptRule(
                        keyword=keyword,
                        rule_text=rule_text,
                        section=section,
                        priority=priority,
                        tags=tags
                    )
                    
                    # Index by keyword
                    if keyword not in self.keyword_to_rules:
                        self.keyword_to_rules[keyword] = []
                    self.keyword_to_rules[keyword].append(rule)
                    
                    # Index by section
                    if section not in self.section_rules:
                        self.section_rules[section] = []
                    self.section_rules[section].append(rule)
                    
                    # Map aliases to keyword (allow many-to-many mapping)
                    keyword_lower = keyword.lower()
                    if keyword_lower not in self.alias_to_keywords:
                        self.alias_to_keywords[keyword_lower] = []
                    if keyword not in self.alias_to_keywords[keyword_lower]:
                        self.alias_to_keywords[keyword_lower].append(keyword)
                    
                    for alias in aliases:
                        alias_lower = alias.lower()
                        if alias_lower not in self.alias_to_keywords:
                            self.alias_to_keywords[alias_lower] = []
                        if keyword not in self.alias_to_keywords[alias_lower]:
                            self.alias_to_keywords[alias_lower].append(keyword)
            
            total_keywords = len(self.keyword_to_rules)
            total_aliases = len(self.alias_to_keywords)
            total_rules = sum(len(r) for r in self.keyword_to_rules.values())
            total_sections = len(self.section_rules)
            
            # logger.info(f"   ✅ Loaded {total_keywords} keywords")
            # logger.info(f"   ✅ Loaded {total_aliases} aliases")
            # logger.info(f"   ✅ Loaded {total_rules} total rules")
            # logger.info(f"   ✅ Organized into {total_sections} sections")
        
        except Exception as e:
            logger.error(f"Error loading rules from JSON: {e}")
    
    def find_matching_keywords(self, query_keyword: str, k: int = 5, threshold: float = 0.2) -> List[Tuple[str, float]]:
        """
        Find keywords that match the query keyword using vector retrieval.
        
        Args:
            query_keyword: The keyword to search for (from user query)
            k: Number of top matches to return
            threshold: Minimum similarity score to include
            
        Returns:
            List of (canonical_keyword, similarity_score) tuples
        """
        matches = []
        
        if self.rule_retriever:
            try:
                # Use vector search to find similar keywords
                results = self.rule_retriever.retrieve_rules(query_keyword, k=k)
                
                for result in results:
                    # Extract the canonical keyword from result
                    canonical_keyword = result.metadata.get('keyword', result.id)
                    
                    if result.score >= threshold:
                        matches.append((canonical_keyword, result.score))
                        logger.debug(f"  🔍 Vector match: '{query_keyword}' -> '{canonical_keyword}' (score: {result.score:.3f})")
                
            except Exception as e:
                logger.debug(f"Vector search failed for '{query_keyword}': {e}")
        
        # Also check alias mappings (now returns list of keywords)
        query_lower = query_keyword.lower()
        if query_lower in self.alias_to_keywords:
            canonical_keywords = self.alias_to_keywords[query_lower]
            for canonical in canonical_keywords:
                # Add if not already in matches
                if not any(m[0] == canonical for m in matches):
                    matches.append((canonical, 1.0))
                    # logger.debug(f"  ✅ Alias match: '{query_keyword}' -> '{canonical}' (score: 1.0)")
        
        # Sort by score descending
        matches.sort(key=lambda x: -x[1])
        return matches[:k]
    
    def get_rules_for_keyword(self, keyword: str) -> List[PromptRule]:
        """Get all rules associated with a keyword."""
        return self.keyword_to_rules.get(keyword, [])
    
    def extract_keywords_from_query_text(self, query: str) -> List[str]:
        """
        Extract keywords by direct string matching on aliases in the query.
        This is a fallback/enhancement to vector-based extraction.
        
        Args:
            query: User's natural language query
            
        Returns:
            List of canonical keywords found via alias matching
        """
        query_lower = query.lower()
        matched_keywords = set()
        
        # Check all aliases (sorted by length descending to match longer phrases first)
        sorted_aliases = sorted(self.alias_to_keywords.items(), key=lambda x: len(x[0]), reverse=True)
        
        for alias, canonical_keywords in sorted_aliases:
            # Check if alias appears in the query
            if alias in query_lower:
                # Add ALL keywords that match this alias
                for kw in canonical_keywords:
                    matched_keywords.add(kw)
                    logger.debug(f"Direct match: '{alias}' -> '{kw}'")
        
        return list(matched_keywords)
    
    def get_rules_by_section(self, section: str) -> List[PromptRule]:
        """Get all rules in a section."""
        return self.section_rules.get(section, [])
    
    def get_rules_for_query_keywords(self, query_keywords: List[str], k: int = 5, 
                                     threshold: float = 0.2) -> Dict[str, List[PromptRule]]:
        """
        Get rules for all query keywords using vector retrieval.
        
        Process:
        1. For each keyword in query_keywords
        2. Find matching keywords using vector retrieval or alias lookup
        3. Get all rules for matched keywords
        4. Organize rules by section
        5. Remove duplicates and sort by priority
        
        Returns:
            Dict mapping section -> list of PromptRule objects
        """
        section_rules: Dict[str, List[PromptRule]] = {}
        seen_rules = set()  # Track unique rules by rule_text
        
        # logger.info(f"\n🔍 DYNAMIC SECTION RETRIEVAL FOR {len(query_keywords)} KEYWORDS")
        # logger.info(f"{'='*80}")
        
        # Normalize threshold using default minimum if caller provided a looser one
        effective_threshold = max(threshold, getattr(self, "default_min_similarity", threshold))
        # Prepare lowercase keyword set for tag overlap checks
        kw_set = {kw.lower() for kw in query_keywords}

        for query_keyword in query_keywords:
            # logger.info(f"\n📌 Processing keyword: '{query_keyword}'")
            
            # Step 1: Find matching keywords using vector retrieval or aliases
            matching_keywords = self.find_matching_keywords(query_keyword, k=k, threshold=effective_threshold)
            
            if not matching_keywords:
                logger.info(f"   ⚠️  No matches found for '{query_keyword}'")
                continue
            
            # logger.info(f"   ✅ Found {len(matching_keywords)} matching keyword(s)")
            
            # Step 2: For each matched keyword, get all associated rules
            for canonical_keyword, match_score in matching_keywords:
                rules = self.get_rules_for_keyword(canonical_keyword)
                
                if rules:
                    # logger.info(f"      📚 '{canonical_keyword}': {len(rules)} rule(s) (match score: {match_score:.3f})")
                    
                    for rule in rules:
                        section = rule.section
                        
                        # Avoid duplicate rules
                        rule_signature = (section, rule.rule_text)
                        if rule_signature in seen_rules:
                            logger.debug(f"Skipping duplicate rule in '{section}'")
                            continue
                        
                        # Secondary relevance filter: require tag overlap with extracted keywords
                        # unless the rule is highly relevant by score or high priority
                        tags_lower = {t.lower() for t in (rule.tags or [])}
                        has_tag_overlap = bool(tags_lower & kw_set)
                        bypass_tag_filter = (match_score >= effective_threshold + 0.15) or (rule.priority >= 8)
                        if tags_lower and not has_tag_overlap and not bypass_tag_filter:
                            logger.debug(f"🚫 Skipping rule due to tag mismatch in '{section}'")
                            continue

                        seen_rules.add(rule_signature)
                        
                        # Add rule to section
                        if section not in section_rules:
                            section_rules[section] = []
                        
                        # Update score to reflect match quality
                        rule_copy = PromptRule(
                            keyword=rule.keyword,
                            rule_text=rule.rule_text,
                            section=rule.section,
                            priority=rule.priority,
                            tags=rule.tags,
                            score=match_score  # Use vector match score
                        )
                        section_rules[section].append(rule_copy)
        
        # Step 3: Sort rules within each section by priority and score
        for section in section_rules:
            section_rules[section].sort(key=lambda r: (-r.priority, -r.score, r.keyword))

        # Step 4: Apply per-section caps to reduce unnecessary/low-value rules
        total_rules_before_caps = sum(len(r) for r in section_rules.values())
        for section, rules in list(section_rules.items()):
            cap = self.selection_caps.get(section, 4)
            if len(rules) > cap:
                section_rules[section] = rules[:cap]

        total_rules_after_caps = sum(len(r) for r in section_rules.values())
        
        return section_rules


# ============================================================================
# Dynamic Prompt Builder
# ============================================================================

class DynamicPromptBuilder:
    """Main class for building dynamic prompts using JSON configuration."""
    
    def __init__(self):
        """Initialize extractor, rules database, and load system prompt from JSON."""
        self.extractor = KeywordExtractor()
        self.rules_db = PromptRulesDB()
        self.template_path = PROMPT_TEMPLATE_PATH
        self.system_prompt = self._load_system_prompt()
        self.footer_prompt = self._load_footer_prompt()
    
    def _load_system_prompt(self) -> str:
        """Load system prompt from prompt_template.json configuration file."""
        try:
            if not PROMPT_TEMPLATE_PATH.exists():
                logger.warning(f"Prompt template file not found: {PROMPT_TEMPLATE_PATH}")
                return self._get_default_system_prompt()
            
            with open(PROMPT_TEMPLATE_PATH, 'r', encoding='utf-8') as f:
                template = json.load(f)
            
            system_prompt = template.get('system_prompt', '')
            if not system_prompt:
                logger.warning("No system_prompt found in template file")
                return self._get_default_system_prompt()
            
            logger.info("System prompt loaded from JSON template")
            return system_prompt
        
        except Exception as e:
            logger.error(f"Error loading system prompt from JSON: {e}")
            return self._get_default_system_prompt()
    
    def _load_json(self, file_path: Path) -> dict:
        """
        Load JSON file with error handling and UTF-8 encoding.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Parsed JSON as dict, or empty dict if error
        """
        try:
            if not file_path.exists():
                logger.warning(f"JSON file not found: {file_path}")
                return {}
            
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        except Exception as e:
            logger.error(f"Error loading JSON from {file_path}: {e}")
            return {}
    
    def _load_footer_prompt(self) -> str:
        """Load footer prompt from prompt_template.json configuration file."""
        try:
            if not PROMPT_TEMPLATE_PATH.exists():
                logger.warning(f"Prompt template file not found: {PROMPT_TEMPLATE_PATH}")
                return self._get_default_footer_prompt()
            
            with open(PROMPT_TEMPLATE_PATH, 'r', encoding='utf-8') as f:
                template = json.load(f)
            
            footer_prompt = template.get('footer_prompt', '')
            if not footer_prompt:
                logger.warning("No footer_prompt found in template file")
                return self._get_default_footer_prompt()
            
            logger.info("Footer prompt loaded from JSON template")
            return footer_prompt
        
        except Exception as e:
            logger.warning(f"Error loading footer prompt: {e}")
            return self._get_default_footer_prompt()
    
    def _get_default_footer_prompt(self) -> str:
        """Return default footer prompt if JSON is not available."""
        return "Generate only one valid Oracle SQL that follows ALL the rules above. Return ONLY the SQL query (no explanations, no markdown, and no commentary)."
    
    def _get_default_system_prompt(self) -> str:
        """Return default system prompt if JSON is not available."""
        return """You are a SQL expert for banking MIS reporting.
Generate Oracle SQL queries for the DM_MIS_DETAILS_VW1 table.
Use only the columns and rules specified in this prompt.
Always use ROUND(..., 2) for decimal values."""
    
    def build_prompt(self, user_query: str, use_vector_examples: bool = True) -> DynamicPrompt:
        """
        Build a dynamic prompt customized for the user query using PURE vector retrieval.
        
        Architecture:
          1. Extract keywords from user query using semantic vector search
          2. Add all static sections (always included from template)
          3. For each extracted keyword:
              - Use vector retrieval to find matching keywords/aliases (sentence-transformer cosine similarity)
              - Retrieve all rules associated with matched keywords
           - Add rules to their corresponding sections
        4. Retrieve similar examples using vector search
        5. Order all sections according to prompt_template.json
        6. Return complete dynamic prompt
        
    All dynamic section retrieval uses vector similarity - NO manual string matching.
        
        Args:
            user_query: Natural language query from user
            use_vector_examples: Whether to use vector-based example retrieval (default: True)
            
        Returns:
            DynamicPrompt object with customized sections
        """
        logger.info(f"Building dynamic prompt...")
        
        # ===== STEP 1: LOAD TEMPLATE AND EXTRACT STATIC SECTION NAMES =====
        template = self._load_json(self.template_path)
        static_section_names = template.get("static_sections", [])
        
        
        # ===== STEP 2: BUILD STATIC SECTIONS =====
        static_sections = self._build_static_sections(template, static_section_names, user_query)
        
        # ===== STEP 3: EXTRACT KEYWORDS FROM USER QUERY =====
        vdb = get_vector_db()
        query_embedding = vdb.encode_query(user_query)
        if query_embedding is not None:
            keywords_vector = self.extractor.extract_with_embedding(query_embedding)
        else:
            keywords_vector = self.extractor.extract(user_query)
        
        # ENHANCEMENT: Also extract keywords via direct alias matching
        keywords_direct = self.rules_db.extract_keywords_from_query_text(user_query)
        
        # Combine both methods (union of results)
        keywords = list(set(keywords_vector + keywords_direct))
        
        
        
        # ===== STEP 4: BUILD DYNAMIC SECTIONS USING VECTOR RETRIEVAL =====
        if keywords:
            # Use vector retrieval that maps keywords -> similar keywords/aliases -> rules -> sections
            section_rules = self.rules_db.get_rules_for_query_keywords(
                keywords,
                k=4,            # Slightly lower top-K per keyword to reduce noise
                threshold=0.5  # Stricter minimum similarity score
            )
        else:
            section_rules = {}
            logger.info("⚠️  No keywords extracted, skipping dynamic sections")
        
        # Build sections from vector-matched rules
        dynamic_sections = self._create_sections(section_rules)
        
        # ===== STEP 5: MERGE STATIC + DYNAMIC SECTIONS =====
        merged_sections = {**static_sections, **dynamic_sections}
        
        # Sort sections by their "order" field to maintain proper sequence
        template_sections = template.get('sections', {})
        sections_with_order = []
        sections_without_order = []
        
        for section_name, section_obj in merged_sections.items():
            if section_name in template_sections:
                order = template_sections[section_name].get('order', float('inf'))
                sections_with_order.append((order, section_name, section_obj))
            else:
                sections_without_order.append((section_name, section_obj))
        
        # Sort by order value
        sections_with_order.sort(key=lambda x: x[0])
        
        # Rebuild sections dict in sorted order
        sections = {}
        for order, section_name, section_obj in sections_with_order:
            sections[section_name] = section_obj
        
        # Add sections without defined order at the end
        for section_name, section_obj in sections_without_order:
            sections[section_name] = section_obj
        
        
        
        # ===== STEP 6: ADD EXAMPLES SECTION =====
        if use_vector_examples:
            examples_section = self._create_examples_section(user_query, query_embedding=query_embedding)
            if examples_section.rules:
                sections["examples"] = examples_section
            else:
                logger.info("⚠️  No examples found")
        
        # ===== STEP 7: CREATE DYNAMIC PROMPT =====
        
        # Extract section order information from template
        section_orders = {}
        template_sections = template.get('sections', {})
        for section_name in sections.keys():
            if section_name in template_sections:
                order = template_sections[section_name].get('order', 999)
                section_orders[section_name] = order
                logger.debug(f"   Section '{section_name}' order: {order}")
        
        prompt = DynamicPrompt(
            system_prompt=self.system_prompt,
            footer_prompt=self.footer_prompt,
            sections=sections,
            user_query=user_query,
            keywords=keywords,
            section_orders=section_orders,  # Pass section orders to DynamicPrompt
            metadata={
                "num_keywords": len(keywords),
                "num_rules": sum(len(rules) for rules in section_rules.values()),
                "num_examples": len(sections.get("examples", PromptSection("", "")).rules),
                "sections": list(sections.keys()),
                "static_sections": static_section_names,
                "dynamic_sections": list(dynamic_sections.keys()),
                "retrieval_method": "vector_semantic_search",
                "extracted_at": datetime.now().isoformat()
            }
        )
        
        return prompt
    
    def _build_static_sections(self, template: dict, static_section_names: List[str], user_query: str) -> Dict[str, PromptSection]:
        """
        Build all static sections that are always included in the prompt.
        
        Static sections contain critical rules that apply to EVERY query regardless of keywords.
        Examples: mandatory_rules, forbidden_columns, query_structure, mrl_handling, etc.
        
        Args:
            template: Loaded prompt template JSON
            static_section_names: List of section names to include as static
            
        Returns:
            Dict of PromptSection objects for static sections
        """
        static_sections = {}
        template_sections = template.get('sections', {})
        
        for section_name in static_section_names:
            if section_name in template_sections:
                section_config = template_sections[section_name]
                
                # Extract section metadata
                title = section_config.get('title', section_name.upper())
                description = section_config.get('description', '')
                rules = section_config.get('rules', [])
                
                # Convert rules to PromptRule objects
                prompt_rules = []
                for rule in rules:
                    if rule:  # Skip empty rules
                        if "{query}" in rule:
                            rule = rule.replace("{query}", user_query)
                        prompt_rules.append(PromptRule(
                            keyword="static",
                            rule_text=rule,
                            section=section_name
                        ))
                
                section = PromptSection(
                    name=title,
                    description=description,
                    rules=prompt_rules
                )
                
                static_sections[section_name] = section
                # logger.info(f"  OK] {section_name}: {len(prompt_rules)} rules")
        
        # logger.info(f"[OK] Static sections built: {len(static_sections)} sections")
        return static_sections
    
    def _create_sections(self, section_rules: Dict[str, List[PromptRule]]) -> Dict[str, PromptSection]:
        """
        Create prompt sections from retrieved rules, organized by section name.
        
        Args:
            section_rules: Dict mapping section name -> list of PromptRule objects
                          (Result from PromptRulesDB.get_rules_for_query_keywords())
        
        Returns:
            Dict mapping section name -> PromptSection object
        """
        sections = {}
        
        # Define metadata for known sections
        section_metadata = {
            "aggregation": ("Aggregation & Calculations", "Functions and calculations to use"),
            "time_handling": ("Time Period Handling", "Rules for different time periods"),
            "filter": ("Filtering & WHERE Clause", "Conditions to apply"),
            "selection": ("SELECT Clause", "Columns to select"),
            "grouping": ("GROUP BY Clause", "Grouping dimensions"),
            "balance_type": ("Balance Type Selection", "Rules for selecting balance types"),
            "currency": ("Currency Handling", "Rules for currency types"),
            "mrl_filtering": ("MRL Filtering", "Rules for MRL line filtering"),
            "product_category": ("Product/Category Handling", "Rules for product categories"),
            "dimension": ("Dimension Selection", "Rules for dimension columns"),
            "calculation": ("Calculation Rules", "Rules for calculations and aggregations"),
            "general": ("General Rules", "General SQL generation rules"),
        }
        
        # Create sections from section_rules
        for section_key, rules_list in section_rules.items():
            if rules_list:  # Only create section if it has rules
                # Get metadata, fallback to generic name if not in metadata
                if section_key in section_metadata:
                    section_name, description = section_metadata[section_key]
                else:
                    # Create generic name from section key
                    section_name = section_key.replace('_', ' ').title()
                    description = f"Rules for {section_name.lower()}"
                
                section = PromptSection(
                    name=section_name,
                    description=description,
                    rules=rules_list
                )
                sections[section_key] = section
                logger.debug(f"Created section '{section_key}' with {len(rules_list)} rules")
        
        return sections
    
    def _create_examples_section(self, user_query: str, query_embedding=None) -> PromptSection:
        """
        Create examples section using vector retrieval from prompt library.
        Retrieves most similar examples using semantic similarity.
        """
        example_retriever: Optional[ExampleRetriever] = _retrievers.get('examples') if _retrievers else None
        
        if not example_retriever:
            logger.warning("Example retriever not available")
            return PromptSection(name="Reference Examples", description="", rules=[])
        
        try:
            # Use vector search to find similar examples (prefer precomputed embedding)
            if query_embedding is not None:
                results = example_retriever.retrieve_examples_with_embedding(query_embedding, k=2)
            else:
                results = example_retriever.retrieve_examples(user_query, k=2)

            examples_rules = []
            for i, result in enumerate(results, 1):
                full_text = result.text or ""

                # Extract User question and SQL query from example text
                user_q = ""
                sql = ""

                # Prefer explicit markers
                uq_match = re.search(r"User:\s*(.*)\nSQL:", full_text, re.IGNORECASE | re.DOTALL)
                if uq_match:
                    user_q = uq_match.group(1).strip()
                else:
                    # Fallback: take text before SQL marker
                    parts = full_text.split("SQL:", 1)
                    if parts:
                        user_q = parts[0].replace("User:", "").strip()

                sql_match = re.search(r"SQL:\n(.*)", full_text, re.IGNORECASE | re.DOTALL)
                if sql_match:
                    sql = sql_match.group(1).strip()

                # Build rule text without tags/IDs; include the example's user question and SQL
                rule_text = f"Example {i}:\nUser: {user_q}\nSQL:\n{sql}"

                rule = PromptRule(
                    keyword=f"example_{i}",
                    rule_text=rule_text,
                    section="examples",
                    priority=5,
                    score=result.score
                )
                examples_rules.append(rule)
                logger.debug(f"Added example {i} (score: {result.score:.3f})")

            return PromptSection(
                name="Reference Examples",
                description="Most similar queries as examples. DO NOT copy them directly; adapt to the current query. Do not copy the Time Period or other specifics.",
                rules=examples_rules
            )
        
        except Exception as e:
            logger.error(f"Failed to retrieve examples: {e}")
            return PromptSection(name="Reference Examples", description="", rules=[])


# ============================================================================
# Prompt Renderer & Integration
# ============================================================================

_BUILDER_SINGLETON: Optional[DynamicPromptBuilder] = None


def get_dynamic_prompt_builder() -> DynamicPromptBuilder:
    """Return a shared DynamicPromptBuilder instance to avoid redundant rebuilds."""
    global _BUILDER_SINGLETON
    if _BUILDER_SINGLETON is None:
        _BUILDER_SINGLETON = DynamicPromptBuilder()
    return _BUILDER_SINGLETON


def build_dynamic_prompt_string(user_query: str, use_vector_retrieval: bool = True, mrl_groups: Dict = None) -> Tuple[str, Dict]:
    """
    Build dynamic prompt using vector retrieval and return as string.
    
    Args:
        user_query: User's natural language query
        use_vector_retrieval: Whether to use vector-based retrieval (default: True)
        mrl_groups: Dictionary of MRL groups for placeholder documentation (optional)
    
    Returns:
        Tuple of (prompt_string, metadata_dict)
    """
    builder = get_dynamic_prompt_builder()
    dynamic_prompt = builder.build_prompt(user_query, use_vector_examples=use_vector_retrieval)
    
    prompt_string = dynamic_prompt.render()
    
    # If MRL groups provided, document them in the prompt
    if mrl_groups:
        # Build documentation of available group placeholders (single line, comma-separated)
        group_docs = []
        for group_name, lines in mrl_groups.items():
            placeholder = f"#GROUP_{group_name.upper().replace(' ', '_')}#"
            group_docs.append(f"{placeholder} (for {group_name}: {len(lines)} lines)")
        
        # Use comma-separated format instead of newline for better display in rules
        group_placeholders_doc = ", ".join(group_docs) if group_docs else "No groups defined"
        
        # Replace the placeholder in the prompt
        prompt_string = prompt_string.replace(
            "{json.dumps(group_placeholders, ensure_ascii=False)}",
            group_placeholders_doc
        )
    
    # Inject month and quarter rules if query mentions them
    prompt_string = inject_month_rules_into_prompt(prompt_string, user_query)
    prompt_string = inject_quarter_rules_into_prompt(prompt_string, user_query)
    
    return prompt_string, dynamic_prompt.metadata

def analyze_prompt_efficiency(dynamic_prompt_size: int, static_prompt_size: int) -> Dict:
    """
    Analyze efficiency gains of dynamic prompting.
    
    Returns:
        Dict with efficiency metrics
    """
    reduction = static_prompt_size - dynamic_prompt_size
    reduction_percent = (reduction / static_prompt_size * 100) if static_prompt_size > 0 else 0
    
    return {
        "static_prompt_size": static_prompt_size,
        "dynamic_prompt_size": dynamic_prompt_size,
        "size_reduction": reduction,
        "reduction_percent": round(reduction_percent, 2),
        "token_savings_estimate": int(reduction / 4),  # Rough estimate: 4 chars per token
    }


# ============================================================================
# Testing & Demonstration
# ============================================================================

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("\n" + "="*80)
    print("DYNAMIC PROMPT BUILDER - DEMONSTRATION")
    print("="*80)
    
    builder = DynamicPromptBuilder()
    
    # Test queries
    test_queries = [
        "give me male customer total deposit"
    ]
    
    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"QUERY: {query}")
        print(f"{'='*80}")
        
        # Build dynamic prompt
        dynamic_prompt = builder.build_prompt(query)
        
        # Display analysis
        print(f"\n📊 ANALYSIS:")
        print(f"  Keywords extracted: {dynamic_prompt.keywords}")
        print(f"  Number of sections: {len(dynamic_prompt.sections)}")
        print(f"  Prompt metadata: {json.dumps(dynamic_prompt.metadata, indent=2)}")
        
        # Display sections
        print(f"\n📝 RENDERED PROMPT:")
        print("-" * 80)
        print(dynamic_prompt.render())
        print("-" * 80)

print("\n" + "="*80)
print("Vector Retrieval System Active")
print("="*80)
