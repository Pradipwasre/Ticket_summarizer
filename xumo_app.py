#!/usr/bin/env python3
"""
Xumo Customer Support AI - Main Application Runner

This is the main entry point for the Xumo Customer Support AI application.
"""

import os
import sys
import argparse
import logging
from typing import List, Dict, Any

from rag_components import XumoKnowledgeBase
from ticket_processor import TicketProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_KB_PATH = "./data/knowledge_base/"
DEFAULT_TICKET_PATH = "./data/tickets/sample_tickets.csv"
DEFAULT_OUTPUT_PATH = "./output/"

class XumoSupportApp:
    """Main application for the Xumo Customer Support AI."""
    
    def __init__(self, api_key: str, model_name: str, kb_path: str):
        """
        Initialize the application.
        
        Args:
            api_key: OpenAI API key
            model_name: Name of the LLM model to use
            kb_path: Path to the knowledge base
        """
        self.api_key = api_key
        self.model_name = model_name
        self.kb_path = kb_path
        
        # Initialize components
        self.knowledge_base = XumoKnowledgeBase(api_key, kb_path)
        self.ticket_processor = TicketProcessor(api_key, model_name)
    
    def process_ticket(self, ticket_text: str) -> str:
        """
        Process a single ticket and generate a summary.
        
        Args:
            ticket_text: Raw ticket text
            
        Returns:
            Formatted ticket summary
        """
        logger.info("Processing ticket...")
        
        # Parse ticket into structured data
        ticket_data = self.ticket_processor.parse_ticket_with_llm(ticket_text)
        
        # Get relevant knowledge context
        knowledge_context = ""
        if self.knowledge_base.ensemble_retriever:
            # Create enhanced query
            query = self.knowledge_base.augment_query_with_context(ticket_data)
            
            # Retrieve relevant knowledge
            knowledge_docs = self.knowledge_base.retrieve_relevant_knowledge(query, k=2)
            
            # Format knowledge context
            if knowledge_docs:
                knowledge_parts = []
                for i, doc in enumerate(knowledge_docs):
                    knowledge_parts.append(f"Knowledge Entry #{i+1}:\n{doc.page_content}")
                knowledge_context = "\n\n".join(knowledge_parts)
        
        # Generate summary
        summary = self.ticket_processor.generate_ticket_summary(ticket_data, knowledge_context)
        
        return summary

def read_tickets_from_csv(file_path: str) -> List[str]:
    """
    Read tickets from a CSV file.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        List of ticket texts
    """
    import csv
    
    tickets = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            # Skip header
            next(reader, None)
            for row in reader:
                if row and len(row) > 0:
                    tickets.append(row[0])
    except Exception as e:
        logger.error(f"Error reading CSV file: {e}")
    
    return tickets

def save_summary(summary: str, output_path: str, index: int):
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
        logger.error(f"Error saving summary: {e}")

def process_single_ticket(text: str, api_key: str, model_name: str, kb_path: str) -> str:
    """
    Process a single ticket provided as text.
    
    Args:
        text: Raw ticket text
        api_key: OpenAI API key
        model_name: LLM model name
        kb_path: Path to knowledge base
        
    Returns:
        Formatted ticket summary
    """
    app = XumoSupportApp(api_key, model_name, kb_path)
    return app.process_ticket(text)

def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description="Xumo Customer Support AI")
    parser.add_argument("--api-key", type=str, help="OpenAI API key")
    parser.add_argument("--model", type=str, default="gpt-4", help="LLM model name")
    parser.add_argument("--kb-path", type=str, default=DEFAULT_KB_PATH, help="Path to knowledge base")
    parser.add_argument("--tickets", type=str, default=DEFAULT_TICKET_PATH, help="Path to ticket samples CSV")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_PATH, help="Output directory path")
    parser.add_argument("--single", type=str, help="Process a single ticket from text")
    
    args = parser.parse_args()
    
    # Get API key from arguments or environment variable
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("OpenAI API key must be provided via --api-key argument or OPENAI_API_KEY environment variable")
        sys.exit(1)
    
    # Process a single ticket from text
    if args.single:
        summary = process_single_ticket(args.single, api_key, args.model, args.kb_path)
        print("\n" + summary + "\n")
        sys.exit(0)
    
    # Process tickets from CSV file
    if not os.path.exists(args.tickets):
        logger.error(f"Ticket file {args.tickets} does not exist")
        sys.exit(1)
    
    # Read tickets
    tickets = read_tickets_from_csv(args.tickets)
    if not tickets:
        logger.error("No tickets found in the input file")
        sys.exit(1)
    
    # Initialize application
    app = XumoSupportApp(api_key, args.model, args.kb_path)
    
    # Process each ticket
    for i, ticket_text in enumerate(tickets):
        try:
            logger.info(f"Processing ticket {i+1}/{len(tickets)}")
            summary = app.process_ticket(ticket_text)
            save_summary(summary, args.output, i+1)
        except Exception as e:
            logger.error(f"Error processing ticket {i+1}: {e}")
    
    logger.info("Processing complete")

if __name__ == "__main__":
    main()