# ELL881 Assignment: Improving Knowledge Graph-based RAG System

This project builds upon the work by Soman et al. (2024) to improve a Knowledge Graph-based Retrieval-Augmented Generation (KG-RAG) system for biomedical multiple-choice question answering (MCQ QA). The aim is to enhance the system's performance by implementing various modifications and analyzing their impact.

## Modified Files

Only the following files have been modified:

1. **`run_mcq_qa.py`**: The main script updated to include new modes and logic for context retrieval and LLM prompting.
2. **`utility.py`**: Updated with new functions and modifications to existing ones to support different modes of operation.
3. **`system_prompts.yaml`**: Added new prompts required for the modified modes.

## Modes of Operation

The current modes of functioning are as follows:

- **Mode 0**: Original KG-RAG
- **Mode 1**: JSON-lized the context from KG search
- **Mode 2**: Added prior domain knowledge
- **Mode 3**: Combined Mode 1 & 2
- **Mode 4**: Alternate Pipeline
- **Mode 5**: Mode 4 with granular fallback

## Setup and Running Instructions

Other instructions regarding setup, preparing the vector database, and running evaluations remain the same as provided in the original project documentation.
