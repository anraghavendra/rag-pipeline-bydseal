import os
import json
import re
from typing import List, Dict, Any, Tuple
from openai import OpenAI
import chromadb
from chromadb.config import Settings

class RAGPipeline:
    """
    RAG Pipeline for BYD Seal Q&A System
    
    Implements a facts-first approach with external reviews as fallback.
    Features robust guardrails to prevent hallucinations and ensure safety.
    """
    
    def __init__(self):
        """Initialize the RAG pipeline with OpenAI client and ChromaDB"""
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.model = 'gpt-4o-mini'  # Efficient model for production use
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self.collection = self.chroma_client.get_or_create_collection(
            name="byd_seal_docs",
            metadata={"hnsw:space": "cosine"}
        )
    
    def ask(self, question: str) -> Dict[str, Any]:
        """
        Main entry point for answering questions about BYD Seal.
        
        Implements the facts-first policy with external reviews as fallback.
        Includes comprehensive guardrails for safety and accuracy.
        
        Args:
            question: User's question about BYD Seal
            
        Returns:
            Dictionary containing answer, status, and citations
        """
        try:
            # Step 1: Analyze question and determine search strategy
            search_strategy = self._analyze_question_and_plan_search(question)
            
            if search_strategy == "refuse":
                return {
                    "answer": "I cannot answer this question as it may involve sensitive information like pricing, warranty, or availability that should only come from official facts.",
                    "status": "refused",
                    "citations": []
                }
            
            # Step 2: Always search facts first (following test.md specification)
            facts_queries = self._generate_facts_search_queries(question)
            facts_docs = self._search_with_queries(facts_queries, "byd_seal_facts.md")
            
            # Step 3: Generate and assess facts-based answer
            if facts_docs:
                facts_answer, facts_used_docs = self._generate_answer_with_context(question, facts_docs)
                facts_adequate = self._assess_answer_adequacy(question, facts_answer, facts_used_docs)
                
                if facts_adequate:
                    # Select citations for facts-based answer
                    citation_docs = self._select_citation_docs(facts_answer, facts_used_docs)
                    citations = self._generate_citations(citation_docs)
                    
                    return {
                        "answer": facts_answer,
                        "status": "answered",
                        "citations": citations
                    }
            
            # Step 4: If facts inadequate and question is external-safe, try external sources
            if search_strategy == "external_safe":
                external_queries = self._generate_external_search_queries(question)
                external_docs = self._search_with_queries(external_queries, "byd_seal_external.json")
                
                if external_docs:
                    external_answer, external_used_docs = self._generate_answer_with_context(question, external_docs)
                    external_adequate = self._assess_answer_adequacy(question, external_answer, external_used_docs)
                    
                    if external_adequate:
                        # Select citations for external-based answer
                        citation_docs = self._select_citation_docs(external_answer, external_used_docs)
                        citations = self._generate_citations(citation_docs)
                        
                        return {
                            "answer": external_answer,
                            "status": "answered",
                            "citations": citations
                        }
            
            # Step 5: No adequate answer found
            return {
                "answer": "I couldn't find any relevant information about this question in the available data sources.",
                "status": "no_information_found",
                "citations": []
            }
            
        except Exception as e:
            return {
                "answer": f"I encountered an error while processing your question: {str(e)}",
                "status": "error",
                "citations": []
            }
    
    def _analyze_question_and_plan_search(self, question: str) -> str:
        """
        Analyze question to determine appropriate search strategy.
        
        Implements guardrails to identify and refuse sensitive topics.
        
        Args:
            question: User's question
            
        Returns:
            Search strategy: "refuse", "facts_only", or "external_safe"
        """
        prompt = f"""Analyze this question: "{question}"

Determine the appropriate search strategy:

1. "refuse" - if the question is about sensitive topics like:
   - Pricing, cost, price, how much
   - Warranty, guarantee, coverage
   - Availability, stock, when available, release date
   - Purchasing, buying, where to buy
   - Any financial or commercial information

2. "facts_only" - if the question is about:
   - Technical specifications, features, specs
   - Battery capacity, range, performance
   - Dimensions, size, weight
   - General information that should come from official facts

3. "external_safe" - if the question is about:
   - Reviews, opinions, experiences
   - What reviewers say, what people think
   - Non-sensitive subjective information
   - User experiences and impressions

Choose the most appropriate strategy:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0.1
            )
            
            strategy = response.choices[0].message.content.strip().lower()
            
            # Validate strategy
            if "refuse" in strategy:
                return "refuse"
            elif "facts" in strategy:
                return "facts_only"
            elif "external" in strategy:
                return "external_safe"
            else:
                # Default to facts_only for safety
                return "facts_only"
                
        except Exception:
            # Default to facts_only for safety
            return "facts_only"
    
    def _generate_facts_search_queries(self, question: str) -> List[str]:
        """
        Generate search queries for factual information.
        
        Creates multiple search terms to improve retrieval accuracy.
        
        Args:
            question: User's question
            
        Returns:
            List of search queries
        """
        prompt = f"""Question: "{question}"

Generate 5-7 search terms to find this information in the BYD Seal facts database. Use simple, core terms that would match relevant content.
Focus on the essential information being asked for, not the full question.

IMPORTANT: The facts database is specifically about the BYD Seal, so:
 - If the question mentions "BYD Seal", "BYD", or "Seal", treat it the same as if it didn't mention the car name
 - Focus ONLY on the core concept being asked for (battery capacity, trim levels, range, etc.)
 - Do NOT include "BYD Seal", "BYD", or "Seal" in your search terms
 - The facts database already contains only BYD Seal information

Examples:
"What is the battery capacity of the byd seal?" → battery capacity, kWh, battery size, energy storage
"What is the battery capacity?" → battery capacity, kWh, battery size, energy storage
"What are the trim levels of the BYD Seal?" → trim levels, Design, Premium, Performance, variants
"What are the trim levels?" → trim levels, Design, Premium, Performance, variants
"What is the range of the byd seal?" → driving range, WLTP range, km range, battery range, distance
"What is the range?" → driving range, WLTP range, km range, battery range, distance

Focus on the core concept being asked for, not the exact wording. Use simple terms that would appear in technical specifications.

Your search terms (one per line):"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.3
            )
            
            queries = response.choices[0].message.content.strip().split('\n')
            # Clean and filter queries
            queries = [q.strip() for q in queries if q.strip() and len(q.strip()) > 2]
            return queries[:7]  # Limit to 7 queries
            
        except Exception:
            # Fallback to simple keyword extraction
            words = re.findall(r'\b\w+\b', question.lower())
            return [word for word in words if len(word) > 3][:5]
    
    def _generate_external_search_queries(self, question: str) -> List[str]:
        """
        Generate search queries for external reviews and opinions.
        
        Creates queries focused on finding review content and opinions.
        
        Args:
            question: User's question
            
        Returns:
            List of search queries
        """
        prompt = f"""Question: "{question}"

Generate 5-7 search terms to find relevant reviews and opinions about the BYD Seal.
Focus on terms that would appear in review content, opinions, and experiences.

Examples:
"What do reviewers say about the audio?" → audio system, sound quality, speakers, music
"What do reviewers say about the interior?" → interior quality, cabin, seats, dashboard
"What do reviewers say about the driving experience?" → driving, handling, performance, experience
"What do reviewers say about the design?" → design, styling, appearance, looks

Your search terms (one per line):"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.3
            )
            
            queries = response.choices[0].message.content.strip().split('\n')
            # Clean and filter queries
            queries = [q.strip() for q in queries if q.strip() and len(q.strip()) > 2]
            return queries[:7]  # Limit to 7 queries
            
        except Exception:
            # Fallback to simple keyword extraction
            words = re.findall(r'\b\w+\b', question.lower())
            return [word for word in words if len(word) > 3][:5]
    
    def _search_with_queries(self, queries: List[str], source: str) -> List[Dict]:
        """
        Search database using multiple LLM-generated queries.
        
        Implements semantic search with ChromaDB for robust retrieval.
        
        Args:
            queries: List of search queries
            source: Data source to search ("byd_seal_facts.md" or "byd_seal_external.json")
            
        Returns:
            List of relevant documents with metadata
        """
        all_docs = []
        
        for query in queries:
            try:
                results = self.collection.query(
                    query_texts=[query],
                    n_results=5,
                    where={"source": source}
                )
                
                for i, doc in enumerate(results['documents'][0]):
                    all_docs.append({
                        'content': doc,
                        'source': source,
                        'doc_id': results['metadatas'][0][i].get('doc_id', ''),
                        'chunk_id': results['metadatas'][0][i].get('doc_id', ''),
                        'distance': results['distances'][0][i]
                    })
                    
            except Exception:
                continue
        
        # Remove duplicates and keep best distance for each doc_id
        unique_docs = {}
        for doc in all_docs:
            doc_id = doc['doc_id']
            if doc_id not in unique_docs or doc['distance'] < unique_docs[doc_id]['distance']:
                unique_docs[doc_id] = doc
        
        # Sort by distance and return top results
        sorted_docs = sorted(unique_docs.values(), key=lambda x: x['distance'])
        
        return sorted_docs[:5]
    
    def _generate_answer_with_context(self, question: str, docs: List[Dict]) -> Tuple[str, List[Dict]]:
        """
        Generate answer using retrieved documents as context.
        
        Implements context-based answer generation to prevent hallucinations.
        
        Args:
            question: User's question
            docs: Retrieved documents
            
        Returns:
            Tuple of (answer, used_documents)
        """
        if not docs:
            return "I couldn't find any relevant information.", []
        
        # Build context from documents
        context_parts = []
        used_docs = []
        total_length = 0
        max_context_length = 6000  # Token limit for context
        
        for i, doc in enumerate(docs):
            content = doc['content']
            source = doc['source']
            
            # Truncate content if too long
            if len(content) > 1500:
                content = content[:1500] + "..."
            
            # Simple document formatting
            doc_text = f"Document {doc.get('doc_id', f'{source}_{i}')} (Source: {source}):\n{content}\n"
            
            # Check if adding this document would exceed the limit
            if total_length + len(doc_text) > max_context_length:
                break
                
            context_parts.append(doc_text)
            total_length += len(doc_text)
            used_docs.append(doc)
        
        context = "\n".join(context_parts)
        
        # Generate answer using LLM
        prompt = f"""Based on the following context, answer this question: {question}

Context:
{context}

CRITICAL RULES - YOU MUST FOLLOW THESE:
- ONLY use information that is explicitly stated in the provided context
- DO NOT add any information that is not in the context
- DO NOT make up channel names, reviewer names, or any other details
- DO NOT mention channels like "[Channel Name]" or generic terms like "reviewers say"
- DO NOT mention "several channels" or "reviewers" unless they are specifically named in the context
- For facts: State the information clearly without citations
- For external reviews: Provide DETAILED and COMPREHENSIVE answers. Include specific quotes, observations, and detailed opinions from each channel. Attribute every opinion or observation to the specific channel names that appear in the context. NEVER write generic phrases like "one reviewer" or "a reviewer"; instead, write "According to <Channel Name>" or similar. Only mention channel names that are actually in the context.

ANSWER LENGTH GUIDELINES:
- For facts questions: Provide concise, factual answers
- For external review questions: Provide detailed, comprehensive answers that thoroughly cover all the opinions, observations, and insights from the different channels. Include specific details, quotes, and nuanced perspectives. Aim for thorough coverage of the available information.

IMPORTANT: If the context contains relevant information, provide a clear and accurate answer. Only say "I don't have enough information" if the context truly doesn't contain any relevant information for the question.

Answer:"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1200,  # Increased for more detailed answers
                temperature=0.3
            )
            
            answer = response.choices[0].message.content.strip()
            
            if not answer:
                return "I couldn't generate a proper answer from the available information.", used_docs
            
            return answer, used_docs
            
        except Exception as e:
            return f"I encountered an error while generating the answer: {str(e)}", used_docs

    def _select_citation_docs(self, answer: str, used_docs: List[Dict]) -> List[Dict]:
        """
        Select citation documents - include all used documents.
        
        Ensures all sources used in answer generation are properly cited.
        
        Args:
            answer: Generated answer
            used_docs: Documents used in answer generation
            
        Returns:
            List of documents to cite
        """
        if not used_docs:
            return []

        # Separate facts and external docs
        facts_docs = [d for d in used_docs if d.get('source') == 'byd_seal_facts.md']
        external_docs = [d for d in used_docs if d.get('source') == 'byd_seal_external.json']

        cited_docs = []

        # For facts: include the most relevant facts doc
        if facts_docs:
            best_fact = sorted(facts_docs, key=lambda d: d.get('distance', 1.0))[0]
            cited_docs.append(best_fact)

        # For external reviews: include ALL external docs that were used in answer generation
        # This ensures all sources mentioned in the answer are properly cited
        if external_docs:
            # Sort by distance and include all external docs (up to a reasonable limit)
            sorted_external = sorted(external_docs, key=lambda d: d.get('distance', 1.0))
            # Include up to 5 external sources to avoid overwhelming citations
            cited_docs.extend(sorted_external[:5])

        return cited_docs
    
    def _assess_answer_adequacy(self, question: str, answer: str, docs: List[Dict]) -> bool:
        """
        Assess answer adequacy using only Euclidean distance from semantic search.
        
        Determines if the generated answer is adequate based on retrieved documents.
        
        Args:
            question: User's question
            answer: Generated answer
            docs: Retrieved documents
            
        Returns:
            True if answer is adequate, False otherwise
        """
        if not docs:
            return False
        
        # Determine the source type
        source_type = "facts" if all(doc.get('source') == 'byd_seal_facts.md' for doc in docs) else "external_reviews"
        
        # For external_safe questions, facts-based answers are never adequate
        question_lower = question.lower()
        opinion_keywords = ["review", "reviewer", "youtuber", "youtube", "opinion", "say", "think", "feel", "experience", "what do"]
        if any(keyword in question_lower for keyword in opinion_keywords):
            if source_type == "facts":
                return False  # Facts are never adequate for opinion questions
        
        # Calculate confidence score based on Euclidean distances
        confidence_score = self._calculate_confidence_score(docs, source_type)
        
        # Apply thresholds - facts questions should be easier to answer
        threshold = 0.4 if source_type == "facts" else 0.5
        
        return confidence_score >= threshold
    
    def _calculate_confidence_score(self, docs: List[Dict], source_type: str) -> float:
        """
        Calculate confidence score using only Euclidean distances from semantic search.
        
        Higher confidence = lower distances (better semantic match).
        
        Args:
            docs: Retrieved documents
            source_type: Type of source ("facts" or "external_reviews")
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not docs:
            return 0.0
        
        # Extract distances from retrieved documents
        distances = [doc.get('distance', 1.0) for doc in docs]
        
        # Calculate average distance (lower is better)
        avg_distance = sum(distances) / len(distances)
        
        # Calculate distance variance (lower variance = more consistent results)
        variance = sum((d - avg_distance) ** 2 for d in distances) / len(distances)
        
        # Apply different normalization based on source type
        if source_type == "facts":
            # For facts, be more lenient with distance thresholds
            distance_confidence = max(0, 1 - (avg_distance / 1.5))  # More lenient normalization
            consistency_confidence = max(0, 1 - (variance * 1.5))   # More lenient variance
        else:
            # For external reviews, keep original thresholds
            distance_confidence = max(0, 1 - (avg_distance / 2.0))  # Original normalization
            consistency_confidence = max(0, 1 - (variance * 2))     # Original variance
        
        # Weighted combination: 70% distance, 30% consistency
        confidence_score = 0.7 * distance_confidence + 0.3 * consistency_confidence
        
        return min(confidence_score, 1.0)  # Cap at 1.0
    
    def _generate_citations(self, docs: List[Dict]) -> List[Dict]:
        """
        Generate citations from retrieved documents with rich metadata.
        
        Extracts metadata for proper source attribution.
        
        Args:
            docs: Documents to generate citations for
            
        Returns:
            List of citation objects with metadata
        """
        citations = []
        for doc in docs:
            source = doc.get('source', 'Unknown')
            doc_id = doc.get('doc_id', '')
            chunk_id = doc.get('chunk_id', '')
            
            # Extract rich metadata for external sources
            if source == 'byd_seal_external.json':
                try:
                    # Parse the content to extract metadata
                    content = doc.get('content', '')
                    
                    # Extract title (only the title, not description or transcript)
                    title = ""
                    if 'Title: ' in content:
                        title_start = content.find('Title: ') + 7
                        # Look for the end of title (before Description: or Transcript:)
                        title_end = content.find(' Description:', title_start)
                        if title_end == -1:
                            title_end = content.find(' Transcript:', title_start)
                        if title_end == -1:
                            title_end = content.find('\n', title_start)
                        if title_end == -1:
                            title_end = len(content)
                        title = content[title_start:title_end].strip()
                        
                        # If title is too long, it might include transcript - truncate it
                        if len(title) > 200:
                            # Find the last reasonable break point
                            break_point = title.rfind(' ', 0, 200)
                            if break_point > 0:
                                title = title[:break_point] + "..."
                    
                    # Extract channel (just the channel name, not the full metadata)
                    channel = ""
                    if 'Channel: ' in content:
                        channel_start = content.find('Channel: ') + 9
                        channel_end = content.find(' Views:', channel_start)
                        if channel_end == -1:
                            channel_end = content.find('\n', channel_start)
                        if channel_end == -1:
                            channel_end = len(content)
                        channel = content[channel_start:channel_end].strip()
                    
                    # Extract views
                    views = ""
                    if 'Views: ' in content:
                        views_start = content.find('Views: ') + 7
                        views_end = content.find(' ', views_start)
                        if views_end == -1:
                            views_end = content.find('\n', views_start)
                        if views_end == -1:
                            views_end = len(content)
                        views = content[views_start:views_end].strip()
                    
                    # Extract subscribers
                    subscribers = ""
                    if 'Channel Subscribers: ' in content:
                        subs_start = content.find('Channel Subscribers: ') + 20
                        subs_end = content.find(' ', subs_start)
                        if subs_end == -1:
                            subs_end = content.find('\n', subs_start)
                        if subs_end == -1:
                            subs_end = len(content)
                        subscribers = content[subs_start:subs_end].strip()
                    
                    citations.append({
                        "source": source,
                        "doc_id": doc_id,
                        "chunk_id": chunk_id,
                        "title": title,
                        "channel": channel,
                        "views": views,
                        "subscribers": subscribers,
                        "type": "external_review"
                    })
                    
                except Exception:
                    # Fallback to basic citation
                    citations.append({
                        "source": source,
                        "doc_id": doc_id,
                        "chunk_id": chunk_id,
                        "type": "external_review"
                    })
            else:
                # Facts citation
                citations.append({
                    "source": source,
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "type": "facts"
                })
        
        return citations