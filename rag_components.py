"""
Xumo Customer Support AI - RAG Components Module

This module implements the Retrieval Augmented Generation components
that enhance the LLM's capabilities with domain-specific knowledge.
"""

import os
import logging
from typing import List, Dict, Any, Optional

from langchain.vectorstores import FAISS  
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.chains import RetrievalQA

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_openai import OpenAIEmbeddings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class XumoKnowledgeBase:
    """
    Implementation of the Retrieval Augmented Generation system for 
    the Xumo customer support knowledge base.
    """
    
    def __init__(self, api_key: str, kb_dir: str):
        """
        Initialize the knowledge base.
        
        Args:
            api_key: OpenAI API key for embeddings
            kb_dir: Directory containing knowledge base documents
        """
        self.api_key = api_key
        self.kb_dir = kb_dir
        self.embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        self.vector_store = None
        self.bm25_retriever = None
        self.ensemble_retriever = None
        
        # Load knowledge base if the directory exists
        if os.path.exists(kb_dir):
            self._load_or_create_knowledge_base()
        else:
            logger.warning(f"Knowledge base directory {kb_dir} does not exist")
    
    def _load_or_create_knowledge_base(self):
        """Load existing knowledge base or create a new one from documents."""
        index_path = os.path.join(self.kb_dir, "faiss_index")
        
        # Check if the index exists
        if os.path.exists(index_path):
            try:
                logger.info("Loading existing vector store...")
                self.vector_store = FAISS.load_local(
                    index_path,
                    self.embeddings
                )
                logger.info("Vector store loaded successfully")
            except Exception as e:
                logger.error(f"Error loading vector store: {e}")
                logger.info("Creating new vector store from documents...")
                self._create_knowledge_base()
        else:
            logger.info("Creating new vector store from documents...")
            self._create_knowledge_base()
        
        # Initialize BM25 retriever for keyword-based retrieval
        docs = self._load_documents()
        if docs:
            self.bm25_retriever = BM25Retriever.from_documents(docs)
            self.bm25_retriever.k = 3
            
            # Create ensemble retriever combining vector and keyword search
            if self.vector_store:
                self.ensemble_retriever = EnsembleRetriever(
                    retrievers=[
                        self.vector_store.as_retriever(search_kwargs={"k": 3}),
                        self.bm25_retriever
                    ],
                    weights=[0.7, 0.3]
                )
    
    def _load_documents(self) -> List[Document]:
        """
        Load documents from the knowledge base directory.
        
        Returns:
            List of Document objects
        """
        documents = []
        for root, _, files in os.walk(self.kb_dir):
            for file in files:
                if file.endswith('.txt') or file.endswith('.md'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            # Extract category from directory structure
                            relative_path = os.path.relpath(file_path, self.kb_dir)
                            parts = relative_path.split(os.sep)
                            category = parts[0] if len(parts) > 1 else "general"
                            
                            # Create document with metadata
                            doc = Document(
                                page_content=content,
                                metadata={
                                    "source": file_path,
                                    "category": category,
                                    "filename": file
                                }
                            )
                            documents.append(doc)
                    except Exception as e:
                        logger.error(f"Error reading file {file_path}: {e}")
        
        return documents
    
    def _create_knowledge_base(self):
        """Create a new vector store from the knowledge base documents."""
        documents = self._load_documents()
        if not documents:
            logger.warning("No documents found in knowledge base directory")
            return
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        split_docs = text_splitter.split_documents(documents)
        
        if not split_docs:
            logger.warning("No document chunks created")
            return
        
        # Create vector store
        try:
            self.vector_store = FAISS.from_documents(split_docs, self.embeddings)
            
            # Save the vector store
            os.makedirs(os.path.join(self.kb_dir, "faiss_index"), exist_ok=True)
            self.vector_store.save_local(os.path.join(self.kb_dir, "faiss_index"))
            
            logger.info(f"Created vector store with {len(split_docs)} chunks")
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
    
    def retrieve_relevant_knowledge(self, query: str, k: int = 3) -> List[Document]:
        """
        Retrieve relevant knowledge base entries for a query.
        
        Args:
            query: The search query
            k: Number of documents to retrieve
            
        Returns:
            List of relevant documents
        """
        if not self.ensemble_retriever:
            logger.warning("Ensemble retriever not initialized")
            return []
        
        try:
            docs = self.ensemble_retriever.get_relevant_documents(query)
            logger.info(f"Retrieved {len(docs)} relevant documents")
            return docs
        except Exception as e:
            logger.error(f"Error retrieving relevant documents: {e}")
            return []
    
    def search_by_category(self, query: str, category: str, k: int = 3) -> List[Document]:
        """
        Search for documents within a specific category.
        
        Args:
            query: The search query
            category: Category to search in
            k: Number of documents to retrieve
            
        Returns:
            List of relevant documents
        """
        if not self.vector_store:
            logger.warning("Vector store not initialized")
            return []
        
        try:
            # Use metadata filtering to search within category
            filter_dict = {"category": category}
            docs = self.vector_store.similarity_search(
                query,
                k=k,
                filter=filter_dict
            )
            logger.info(f"Retrieved {len(docs)} documents from category '{category}'")
            return docs
        except Exception as e:
            logger.error(f"Error searching by category: {e}")
            return []
    
    def search_by_device_type(self, query: str, device_type: str, k: int = 3) -> List[Document]:
        """
        Search for knowledge base entries relevant to a specific device type.
        
        Args:
            query: The search query
            device_type: Device type (e.g., "Xumo TV", "Xumo Stream Box")
            k: Number of documents to retrieve
            
        Returns:
            List of relevant documents
        """
        # Create a more specific query that incorporates the device type
        enhanced_query = f"{device_type} {query}"
        
        if not self.vector_store:
            logger.warning("Vector store not initialized")
            return []
        
        try:
            # First try exact filter match if metadata includes device type
            filter_docs = []
            try:
                filter_docs = self.vector_store.similarity_search(
                    query,
                    k=k,
                    filter={"device_type": device_type}
                )
            except:
                # If filtering fails, just continue with the enhanced query
                pass
                
            # If we got results with filtering, return them
            if filter_docs:
                return filter_docs
                
            # Otherwise use the enhanced query
            docs = self.vector_store.similarity_search(enhanced_query, k=k)
            logger.info(f"Retrieved {len(docs)} documents for device type '{device_type}'")
            return docs
        except Exception as e:
            logger.error(f"Error searching by device type: {e}")
            return []
    
    def augment_query_with_context(self, ticket_data: Dict[str, Any]) -> str:
        """
        Create an enhanced search query from ticket data to improve retrieval.
        
        Args:
            ticket_data: Structured ticket data
            
        Returns:
            Enhanced query string
        """
        # Extract relevant fields
        device_type = ticket_data.get("device_type", "")
        short_description = ticket_data.get("short_description", "")
        issue_description = ticket_data.get("brief_description", "")
        troubleshooting = ticket_data.get("troubleshooting_steps", "")
        status = ticket_data.get("status", "")
        
        # Construct enhanced query
        query_parts = []
        if device_type:
            query_parts.append(f"Device: {device_type}")
        if short_description:
            query_parts.append(f"Issue: {short_description}")
        if issue_description:
            query_parts.append(f"Description: {issue_description}")
        
        # Add troubleshooting steps that failed, as these are important for context
        if troubleshooting and "persist" in troubleshooting.lower():
            query_parts.append(f"Failed troubleshooting: {troubleshooting}")
        
        # Create the final query
        query = " ".join(query_parts)
        return query
        
    def get_knowledge_context(self, ticket_data: Dict[str, Any], max_docs: int = 3) -> str:
        """
        Get formatted knowledge context for a ticket.
        
        Args:
            ticket_data: Structured ticket data
            max_docs: Maximum number of documents to include
            
        Returns:
            Formatted knowledge context string
        """
        device_type = ticket_data.get("device_type", "")
        
        # Create enhanced query
        query = self.augment_query_with_context(ticket_data)
        
        # Retrieve relevant documents
        docs = []
        if device_type:
            # First try device-specific search
            docs = self.search_by_device_type(query, device_type, k=max_docs)
        
        # If device-specific search didn't yield enough results, use general search
        if len(docs) < max_docs:
            general_docs = self.retrieve_relevant_knowledge(
                query, 
                k=(max_docs - len(docs))
            )
            docs.extend(general_docs)
        
        # Format the context
        if not docs:
            return "No relevant knowledge base entries found."
        
        context_parts = []
        for i, doc in enumerate(docs[:max_docs]):
            source = doc.metadata.get("source", "Unknown source")
            filename = os.path.basename(source)
            context_parts.append(f"Knowledge Entry #{i+1} (from {filename}):\n{doc.page_content}")
        
        return "\n\n".join(context_parts)