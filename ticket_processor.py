"""
Xumo Customer Support AI - Ticket Processing Module

This module handles the parsing and structuring of raw customer support tickets.
"""

import re
import logging
from typing import Dict, Any, List, Optional

from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TicketProcessor:
    """
    Processes raw customer support tickets into structured data and summaries.
    """
    
    def __init__(self, api_key: str, model_name: str = "gpt-4"):
        """
        Initialize the ticket processor.
        
        Args:
            api_key: OpenAI API key
            model_name: LLM model to use
        """
        self.api_key = api_key
        self.model_name = model_name
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            openai_api_key=api_key,
            model_name=model_name,
            temperature=0.1
        )
    
    def extract_mac_address(self, text: str) -> Optional[str]:
        """
        Extract MAC address from text using regex.
        
        Args:
            text: Text to search for MAC address
            
        Returns:
            MAC address if found, None otherwise
        """
        # Pattern for MAC address (various formats)
        patterns = [
            r'(?i)MAC\s*(?:Address)?[:;]?\s*([0-9A-F]{2}[:.-][0-9A-F]{2}[:.-][0-9A-F]{2}[:.-][0-9A-F]{2}[:.-][0-9A-F]{2}[:.-][0-9A-F]{2})',
            r'(?i)MAC\s*(?:Address)?[:;]?\s*([0-9A-F]{12})',
            r'(?i)([0-9A-F]{2}[:.-][0-9A-F]{2}[:.-][0-9A-F]{2}[:.-][0-9A-F]{2}[:.-][0-9A-F]{2}[:.-][0-9A-F]{2})'
        ]
        
        for pattern in patterns:
            matches = re.search(pattern, text)
            if matches:
                return matches.group(1)
        
        return None
    
    def extract_serial_number(self, text: str) -> Optional[str]:
        """
        Extract Xumo serial number from text using regex.
        
        Args:
            text: Text to search for serial number
            
        Returns:
            Serial number if found, None otherwise
        """
        # Xumo serial numbers start with XI or ES and have 16 alphanumeric characters
        patterns = [
            r'(?i)Serial\s*(?:Number)?[:;]?\s*(XI[A-Z0-9]{14})',
            r'(?i)Serial\s*(?:Number)?[:;]?\s*(ES[A-Z0-9]{14})'
        ]
        
        for pattern in patterns:
            matches = re.search(pattern, text)
            if matches:
                return matches.group(1)
        
        return None
    
    def extract_device_type(self, text: str) -> Optional[str]:
        """
        Extract device type from text.
        
        Args:
            text: Text to search for device type
            
        Returns:
            Device type if found, None otherwise
        """
        if re.search(r'(?i)Stream\s*Box', text):
            return "Xumo Stream Box"
        elif re.search(r'(?i)Xumo\s*TV', text):
            return "Xumo TV"
        
        return None
    
    def extract_isp(self, text: str) -> Optional[str]:
        """
        Extract internet service provider from text.
        
        Args:
            text: Text to search for ISP
            
        Returns:
            ISP if found, None otherwise
        """
        isp_patterns = [
            r'(?i)Internet\s*(?:Service)?\s*Provider[:;]?\s*([A-Za-z]+)',
            r'(?i)ISP[:;]?\s*([A-Za-z]+)'
        ]
        
        common_isps = [
            'Spectrum', 'Comcast', 'Xfinity', 'AT&T', 'Verizon', 'Cox',
            'CenturyLink', 'Frontier', 'Optimum', 'Suddenlink', 'Mediacom',
            'WOW', 'Windstream', 'HughesNet', 'ViaSat', 'Starlink'
        ]
        
        # Try to find ISP using patterns
        for pattern in isp_patterns:
            matches = re.search(pattern, text)
            if matches:
                return matches.group(1)
        
        # If patterns fail, check for known ISP names
        for isp in common_isps:
            if re.search(r'\b' + re.escape(isp) + r'\b', text, re.IGNORECASE):
                return isp
        
        return None
    
    def extract_status(self, text: str) -> str:
        """
        Extract ticket status from text.
        
        Args:
            text: Text to search for status
            
        Returns:
            Status string
        """
        if re.search(r'(?i)Escalated\s*to\s*Tier\s*2', text):
            return "Escalated to Tier 2"
        elif re.search(r'(?i)Escalated', text):
            return "Escalated"
        elif re.search(r'(?i)Resolved', text):
            return "Resolved"
        else:
            return "Not Resolved"
    
    def extract_ticket_data_rule_based(self, text: str) -> Dict[str, Any]:
        """
        Extract ticket data using rule-based methods.
        
        Args:
            text: Raw ticket text
            
        Returns:
            Dictionary of extracted data
        """
        # Extract basic information using regex
        mac_address = self.extract_mac_address(text)
        serial_number = self.extract_serial_number(text)
        device_type = self.extract_device_type(text)
        isp = self.extract_isp(text)
        status = self.extract_status(text)
        
        # Extract customer name
        name_match = re.search(r'(?i)Name[:;]?\s*([A-Za-z\s]+)', text)
        customer_name = name_match.group(1).strip() if name_match else None
        
        # Extract contact number
        phone_match = re.search(r'(?i)(?:Contact|Phone|Number)[:;]?\s*([\d\-\(\)\s]+)', text)
        contact_number = phone_match.group(1).strip() if phone_match else None
        
        # Extract short description
        short_desc_match = re.search(r'(?i)Short\s*Description[:;]?\s*([^\n]+)', text)
        short_description = short_desc_match.group(1).strip() if short_desc_match else None
        
        return {
            "mac_address": mac_address,
            "serial_number": serial_number,
            "device_type": device_type,
            "isp": isp,
            "status": status,
            "customer_name": customer_name,
            "contact_number": contact_number,
            "short_description": short_description
        }
    
    def parse_ticket_with_llm(self, ticket_text: str) -> Dict[str, Any]:
        """
        Parse ticket text into structured data using LLM.
        
        Args:
            ticket_text: Raw ticket text
            
        Returns:
            Dictionary of structured ticket data
        """
        # First try rule-based extraction for key fields
        rule_based_data = self.extract_ticket_data_rule_based(ticket_text)
        
        # Use LLM to extract remaining fields and improve extraction
        extraction_template = """
        You are an AI expert in Xumo customer support ticket analysis. Extract the following information from the ticket below:
        
        1. Short issue description (one line)
        2. Brief issue description (2-3 sentences)
        3. Troubleshooting steps performed (comma-separated list)
        4. Current status
        
        Some information has already been extracted:
        - Device Type: {device_type}
        - MAC Address: {mac_address}
        - Serial Number: {serial_number}
        - ISP: {isp}
        
        Ticket:
        {ticket_text}
        
        Return ONLY a JSON object with these fields:
        - short_description: one-line summary of the core issue
        - brief_description: 2-3 sentence description of the issue
        - troubleshooting_steps: comma-separated list of troubleshooting actions
        - status: current ticket status (Resolved, Not Resolved, or Escalated to Tier X)
        
        No explanations or additional text.
        """
        
        extraction_prompt = PromptTemplate(
            input_variables=["ticket_text", "device_type", "mac_address", "serial_number", "isp"],
            template=extraction_template
        )
        
        extraction_chain = LLMChain(llm=self.llm, prompt=extraction_prompt)
        
        try:
            result = extraction_chain.run(
                ticket_text=ticket_text,
                device_type=rule_based_data.get("device_type", "Unknown"),
                mac_address=rule_based_data.get("mac_address", "Unknown"),
                serial_number=rule_based_data.get("serial_number", "Unknown"),
                isp=rule_based_data.get("isp", "Unknown")
            )
            
            # Parse the JSON response
            import json
            try:
                llm_data = json.loads(result)
                
                # Combine rule-based and LLM data, with LLM data taking precedence
                combined_data = {**rule_based_data, **llm_data}
                return combined_data
            except json.JSONDecodeError:
                logger.error("Failed to parse LLM response as JSON")
                return rule_based_data
        except Exception as e:
            logger.error(f"Error in LLM extraction: {e}")
            return rule_based_data
    
    def generate_ticket_summary(self, ticket_data: Dict[str, Any], knowledge_context: str = "") -> str:
        """
        Generate a standardized summary from ticket data.
        
        Args:
            ticket_data: Structured ticket data
            knowledge_context: Knowledge base context
            
        Returns:
            Formatted ticket summary
        """
        summary_template = """
        You are an AI assistant for Xumo TV customer support. Generate a concise summary of a customer support ticket.
        
        The summary must follow this exact format:
        
        Ticket Summary
        **Short Issue Description:** [One-line description]
        **Brief Issue Description:** [2-3 sentence description]
        **Troubleshooting Performed:** [Comma-separated list of troubleshooting steps]
        **Status:** [Current status]
        
        **MAC Address:** [MAC address if available]
        **Serial Number:** [Serial number if available]
        **ISP:** [Internet Service Provider if available]
        
        Here's the ticket information:
        {ticket_data}
        
        Here's relevant knowledge from the Xumo knowledge base:
        {knowledge_context}
        
        Generate ONLY the requested format with NO additional text.
        """
        
        summary_prompt = PromptTemplate(
            input_variables=["ticket_data", "knowledge_context"],
            template=summary_template
        )
        
        summary_chain = LLMChain(llm=self.llm, prompt=summary_prompt)
        
        try:
            summary = summary_chain.run(
                ticket_data=str(ticket_data),
                knowledge_context=knowledge_context
            )
            
            return summary.strip()
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            
            # Fallback summary if LLM fails
            fallback = self._generate_fallback_summary(ticket_data)
            return fallback
    
    def _generate_fallback_summary(self, ticket_data: Dict[str, Any]) -> str:
        """
        Generate a fallback summary if LLM processing fails.
        
        Args:
            ticket_data: Structured ticket data
            
        Returns:
            Basic formatted ticket summary
        """
        # Extract available fields with defaults
        short_description = ticket_data.get("short_description", "Unknown Issue")
        brief_description = ticket_data.get("brief_description", "No description available.")
        troubleshooting = ticket_data.get("troubleshooting_steps", "No troubleshooting information available.")
        status = ticket_data.get("status", "Unknown Status")
        mac_address = ticket_data.get("mac_address", "Not provided")
        serial_number = ticket_data.get("serial_number", "Not provided")
        isp = ticket_data.get("isp", "Not provided")
        
        # Format the summary
        summary = f"""Ticket Summary
**Short Issue Description:** {short_description}
**Brief Issue Description:** {brief_description}
**Troubleshooting Performed:** {troubleshooting}
**Status:** {status}

**MAC Address:** {mac_address}
**Serial Number:** {serial_number}
**ISP:** {isp}"""
        
        return summary