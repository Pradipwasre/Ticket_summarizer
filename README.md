# Xumo Customer Support AI

A Python application that leverages Large Language Models (LLMs) and Retrieval Augmented Generation (RAG) to transform detailed customer support tickets into concise, standardized summaries for Xumo TV and Stream Box support.

## Project Overview

The Xumo Customer Support AI system helps support agents manage and standardize customer support tickets by:

1. Processing raw support tickets from various sources
2. Extracting key information (device type, issues, MAC addresses, etc.)
3. Enriching the data with relevant knowledge from the Xumo support database
4. Generating standardized ticket summaries that highlight the most important information

## Features

- **Ticket Parsing**: Extract structured data from freeform text using LLMs and rule-based processing
- **Knowledge Retrieval**: Enrich ticket analysis with relevant support documentation
- **Summary Generation**: Create concise, standardized ticket summaries
- **Batch Processing**: Process multiple tickets from CSV files
- **Single Ticket Processing**: Process individual tickets via command line

## Architecture

The system consists of three main components:

1. **Ticket Processor**: Parses and structures raw support tickets
2. **Knowledge Base**: Manages and retrieves relevant knowledge using vector search
3. **Main Application**: Coordinates processing and generates summaries

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/xumo-support-ai.git
   cd xumo-support-ai
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up your OpenAI API key:
   ```
   export OPENAI_API_KEY="your-api-key-here"
   ```

## Usage

### Process a Single Ticket

```
python app.py --single "Your raw ticket text here"
```

### Process Multiple Tickets from a CSV File

```
python app.py --tickets /path/to/tickets.csv --output /path/to/output/directory
```

### Command Line Arguments

- `--api-key`: OpenAI API key (optional if set as environment variable)
- `--model`: LLM model name (default: gpt-4)
- `--kb-path`: Path to knowledge base directory (default: ./data/knowledge_base/)
- `--tickets`: Path to ticket samples CSV (default: ./data/tickets/sample_tickets.csv)
- `--output`: Output directory path (default: ./output/)
- `--single`: Process a single ticket from text

## Directory Structure

```
xumo-support-ai/
├── app.py                  # Main application runner
├── ticket_processor.py     # Ticket parsing and structuring
├── rag_components.py       # RAG implementation
├── requirements.txt        # Project dependencies
├── data/
│   ├── knowledge_base/     # Xumo support knowledge base
│   │   └── ...
│   └── tickets/            # Sample ticket data
│       └── sample_tickets.csv
└── output/                 # Generated summaries
    └── ...
```

## RAG Implementation

The RAG system enhances the LLM's capabilities by:

1. **Knowledge Retrieval**: Using FAISS vector store for semantic search
2. **Hybrid Search**: Combining vector-based and keyword-based (BM25) search
3. **Context Augmentation**: Enriching prompts with relevant support knowledge
4. **Domain-Specific Knowledge**: Leveraging Xumo-specific troubleshooting information

## Sample Output

```
Ticket Summary
**Short Issue Description:** Stream Box Internet Connection Failure
**Brief Issue Description:** Customer's Xumo Stream Box unable to connect to internet despite functional network and multiple troubleshooting attempts across connection methods.
**Troubleshooting Performed:** Power cycle, factory reset, Ethernet connection attempt, hotspot connection test