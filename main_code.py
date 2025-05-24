#!/usr/bin/env python3
"""
Xumo Customer Support AI - Main Application
This script processes raw customer support tickets and converts them into standardized summaries.
"""

import os
import csv
import argparse
import logging
from typing import Dict, List, Any

from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.retrievers import BM25Retriever, EnsembleRetriever

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_KB_PATH = "data/knowledge_base/"
DEFAULT_TICKET_SAMPLES = "data/tickets/sample_tickets.csv"
DEFAULT_OUTPUT_PATH = "output/"

class XumoSupportAI:
    """Main class for the Xumo customer support ticket summarization system."""
    
    def __init__(self, api_key: str, model_name: str = "gpt-4", kb_path: str = DEFAULT_KB_PATH):
        """
        Initialize the Xumo Support AI system.
        
        Args:
            api_key: OpenAI API key
            model_name: Name of the LLM model to use
            kb_path: Path to the knowledge base documents
        """
        if not api_key:
            raise ValueError("API key must be provided")
        
        self.api_key = api_key
        self.model_name = model_name
        self.kb_path = kb_path
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            openai_api_key=api_key, 
            model_name=model_name,
            temperature=0.1
        )
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        
        # Initialize knowledge base
        self.knowledge_base = None
        self.bm25_retriever = None
        self.ensemble_retriever = None
        
        # Load knowledge base if path exists
        if os.path.exists(kb_path):
            self._load_knowledge_base()
        else:
            logger.warning(f"Knowledge base path {kb_path} does not exist")
    
    def _load_knowledge_base(self):
        """Load and process the knowledge base documents."""
        logger.info("Loading knowledge base documents...")
        
        documents = []
        for root, _, files in os.walk(self.kb_path):
            for file in files:
                if file.endswith('.txt') or file.endswith('.md'):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        documents.append(Document(page_content=content, metadata={"source": file_path}))
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        split_docs = text_splitter.split_documents(documents)
        
        # Create vector store
        self.knowledge_base = FAISS.from_documents(split_docs, self.embeddings)
        
        # Create BM25 retriever for keyword-based retrieval
        self.bm25_retriever = BM25Retriever.from_documents(split_docs)
        self.bm25_retriever.k = 3
        
        # Create ensemble retriever
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.knowledge_base.as_retriever(search_kwargs={"k": 3}), self.bm25_retriever],
            weights=[0.7, 0.3]
        )
        
        logger.info(f"Loaded {len(split_docs)} knowledge base chunks")
    
    def parse_ticket(self, ticket_text: str) -> Dict[str, Any]:
        """
        Parse the raw ticket text into structured data.
        
        Args:
            ticket_text: Raw ticket text
            
        Returns:
            Structured ticket data
        """
        # Use LLM to extract structured data from the ticket text
        extraction_template = """
        Extract the following information from the customer support ticket below:
        
        1. Short description of the issue
        2. Device type (Xumo TV or Xumo Stream Box)
        3. Customer name
        4. Customer contact number
        5. MAC address
        6. Serial number
        7. Brief description of the issue
        8. Troubleshooting steps performed
        9. Current status
        10. Internet service provider (if mentioned)
        
        Ticket:
        {ticket_text}
        
        Provide the extracted information in a JSON format.
        """
        
        extraction_prompt = PromptTemplate(
            input_variables=["ticket_text"],
            template=extraction_template
        )
        
        extraction_chain = LLMChain(llm=self.llm, prompt=extraction_prompt)
        result = extraction_chain.run(ticket_text=ticket_text)
        
        # The result should be JSON, but just in case the LLM doesn't format it correctly
        try:
            import json
            parsed_data = json.loads(result)
            return parsed_data
        except json.JSONDecodeError:
            logger.error("Failed to parse LLM response as JSON")
            # Return a simple dictionary with the raw result
            return {"raw_extraction": result}
    
    def retrieve_knowledge(self, ticket_data: Dict[str, Any]) -> List[Document]:
        """
        Retrieve relevant knowledge base entries for the ticket.
        
        Args:
            ticket_data: Structured ticket data
            
        Returns:
            List of relevant knowledge base documents
        """
        if not self.ensemble_retriever:
            logger.warning("Knowledge base not loaded, skipping retrieval")
            return []
        
        # Create a query from the ticket data
        query = f"""
        Issue: {ticket_data.get('short_description', '')}
        Device: {ticket_data.get('device_type', '')}
        Description: {ticket_data.get('brief_description', '')}
        Troubleshooting: {ticket_data.get('troubleshooting_steps', '')}
        """
        
        # Retrieve relevant documents
        docs = self.ensemble_retriever.get_relevant_documents(query)
        logger.info(f"Retrieved {len(docs)} relevant knowledge base entries")
        
        return docs
    
    def generate_summary(self, ticket_data: Dict[str, Any], knowledge_docs: List[Document]) -> str:
        """
        Generate a concise summary of the ticket.
        
        Args:
            ticket_data: Structured ticket data
            knowledge_docs: Relevant knowledge base documents
            
        Returns:
            Formatted ticket summary
        """
        # Extract relevant knowledge context
        knowledge_context = "\n\n".join([doc.page_content for doc in knowledge_docs[:3]])
        
        # Create the summary prompt
        summary_template = """
        You are an AI assistant for Xumo TV customer support. Your task is to generate a concise summary of a customer support ticket.
        
        The summary should follow this exact format:
        
        Ticket Summary
        **Short Issue Description:** [One-line description of the core issue]
        **Brief Issue Description:** [2-3 sentence description of the issue]
        **Troubleshooting Performed:** [Comma-separated list of troubleshooting steps]
        **Status:** [Current status - Resolved, Not Resolved, or Escalated to Tier X]
        
        **MAC Address:** [MAC address if available]
        **Serial Number:** [Serial number if available]
        **ISP:** [Internet Service Provider if available]
        
        Here's the ticket information:
        {ticket_data}
        
        Here's relevant knowledge from the Xumo knowledge base that might help:
        {knowledge_context}
        
        Generate only the requested summary format without additional text, headers, or explanations.
        """
        
        summary_prompt = PromptTemplate(
            input_variables=["ticket_data", "knowledge_context"],
            template=summary_template
        )
        
        summary_chain = LLMChain(llm=self.llm, prompt=summary_prompt)
        summary = summary_chain.run(
            ticket_data=str(ticket_data),
            knowledge_context=knowledge_context
        )
        
        return summary.strip()
    
    def process_ticket(self, ticket_text: str) -> str:
        """
        Process a raw ticket and generate a standardized summary.
        
        Args:
            ticket_text: Raw ticket text
            
        Returns:
            Standardized ticket summary
        """
        logger.info("Processing ticket...")
        
        # Parse the ticket
        ticket_data = self.parse_ticket(ticket_text)
        
        # Retrieve relevant knowledge
        knowledge_docs = self.retrieve_knowledge(ticket_data)
        
        # Generate summary
        summary = self.generate_summary(ticket_data, knowledge_docs)
        
        return summary


def read_tickets_from_csv(file_path: str) -> List[str]:
    """
    Read ticket data from a CSV file.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        List of ticket texts
    """
    tickets = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip header
            for row in reader:
                if row and len(row) > 0:
                    tickets.append(row[0])  # Assuming ticket text is in the first column
    except Exception as e:
        logger.error(f"Error reading CSV file: {e}")
    
    return tickets


def save_summary_to_file(summary: str, output_path: str, index: int):
    """
    Save a ticket summary to a file.
    
    Args:
        summary: Ticket summary
        output_path: Output directory path
        index: Ticket index
    """
    os.makedirs(output_path, exist_ok=True)
    file_path = os.path.join(output_path, f"ticket_summary_{index}.md")
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(summary)
        logger.info(f"Summary saved to {file_path}")
    except Exception as e:
        logger.error(f"Error saving summary to file: {e}")


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description="Xumo Customer Support AI")
    parser.add_argument("--api-key", type=str, help="OpenAI API key")
    parser.add_argument("--model", type=str, default="gpt-4", help="LLM model to use")
    parser.add_argument("--kb-path", type=str, default=DEFAULT_KB_PATH, help="Path to knowledge base documents")
    parser.add_argument("--tickets", type=str, default=DEFAULT_TICKET_SAMPLES, help="Path to ticket samples CSV")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_PATH, help="Output directory path")
    
    args = parser.parse_args()
    
    # Get API key from args or environment
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("OpenAI API key must be provided via --api-key argument or OPENAI_API_KEY environment variable")
        return
    
    # Initialize the system
    try:
        system = XumoSupportAI(api_key=api_key, model_name=args.model, kb_path=args.kb_path)
    except ValueError as e:
        logger.error(f"Failed to initialize system: {e}")
        return
    
    # Read ticket samples
    if not os.path.exists(args.tickets):
        logger.error(f"Ticket samples file {args.tickets} does not exist")
        return
    
    tickets = read_tickets_from_csv(args.tickets)
    if not tickets:
        logger.error("No tickets found in the samples file")
        return
    
    # Process each ticket
    for i, ticket_text in enumerate(tickets):
        try:
            logger.info(f"Processing ticket {i+1}/{len(tickets)}")
            summary = system.process_ticket(ticket_text)
            save_summary_to_file(summary, args.output, i+1)
        except Exception as e:
            logger.error(f"Error processing ticket {i+1}: {e}")
    
    logger.info("Processing complete")


if __name__ == "__main__":
    main()
