# Xumo Customer Support AI System Architecture

## Project Overview

This system transforms detailed Xumo customer support tickets into concise, standardized summaries using LLMs enhanced with Retrieval Augmented Generation (RAG). The architecture leverages Xumo's knowledge base to improve response quality and ensure domain-specific accuracy.

## Architecture Components

```
┌─────────────────────┐     ┌───────────────────────┐     ┌────────────────────┐
│                     │     │                       │     │                    │
│  Raw Support Ticket │────▶│ Ticket Preprocessor   │────▶│ Vector Database    │
│                     │     │                       │     │                    │
└─────────────────────┘     └───────────────────────┘     └────────────────────┘
                                                                    │
                                                                    ▼
┌─────────────────────┐     ┌───────────────────────┐     ┌────────────────────┐
│                     │     │                       │     │                    │
│  Summary Template   │◀────│ LLM Processor         │◀────│ Knowledge Retriever│
│                     │     │                       │     │                    │
└─────────────────────┘     └───────────────────────┘     └────────────────────┘
                                      │
                                      ▼
                            ┌────────────────────┐
                            │                    │
                            │ Structured Summary │
                            │                    │
                            └────────────────────┘
```

## Key Components

1. **Ticket Preprocessor**: Parses raw support tickets into structured data.
2. **Vector Database**: Stores embeddings of Xumo knowledge base and historical tickets.
3. **Knowledge Retriever**: Pulls relevant context from the Xumo knowledge base.
4. **LLM Processor**: Generates concise summaries with domain-specific understanding.
5. **Summary Template**: Ensures consistent formatting for all ticket summaries.

## Data Flow

1. Support tickets are ingested from Zendesk or Mercury chat logs.
2. Text is preprocessed and key entities (device types, issues, MAC addresses) are extracted.
3. Similar past tickets and relevant knowledge base entries are retrieved.
4. The LLM synthesizes a standardized summary with the RAG-enhanced context.
5. Templates ensure the output follows the required format for downstream systems.

## RAG Implementation

The RAG system enhances the LLM's capabilities by:
- Providing up-to-date Xumo device specifics and troubleshooting protocols
- Retrieving similar historical tickets and their resolutions
- Ensuring domain-specific terminology is used correctly
- Improving accuracy of issue classification

## Deployment

The system is containerized using Docker and can be deployed as a standalone service or integrated with existing support platforms via API endpoints.
