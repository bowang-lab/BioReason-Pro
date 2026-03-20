# Data Processing Pipeline

This folder contains notebooks for processing biological datasets, primarily focused on creating reasoning datasets for protein function prediction using Gene Ontology (GO) annotations.

## Overview

The notebooks follow a pipeline workflow from raw CAFA5 data to enriched reasoning datasets, then to training data generation and evaluation. The main workflow progresses through: **CAFA5 dataset creation** → **Training data generation** → **Test set creation** → **SwissProt processing** → **Benchmarking**.

---

## Phase 1: CAFA5 Dataset Creation & Enhancement

### 001-GO-Dataset.ipynb
- **Purpose**: Creates the initial `cafa5_reasoning` HuggingFace dataset
- **Function**: Parses GO terms from CAFA5 training data
- **Output**: Foundation dataset for all reasoning work

### 002-cafa5_string_interactions.ipynb
- **Purpose**: Adds Protein-Protein Interaction (PPI) metadata from STRING database
- **Function**: Downloads and processes STRING protein links and info files
- **Output**: Enriches CAFA5 dataset with interaction data

### 003-cafa5_ppi_protein_name_cleaning.ipynb
- **Purpose**: Cleans and standardizes protein names in the CAFA5 PPI metadata
- **Function**: Removes EC classifications, cellular location suffixes, and redundant descriptions
- **Output**: Cleaned dataset optimized for LLM training

### 004-GO-Reasoning-Data.ipynb
- **Purpose**: Enhances `cafa5_reasoning` dataset with additional metadata fields
- **Function**: Adds PPI, BLAST, and other biological features
- **Output**: Expanded dataset with richer context for reasoning

### 005-Protein_Structure_and_Domains.ipynb
- **Purpose**: Adds protein structure metadata and domain information
- **Function**: 
  - Integrates AlphaFold structures and PDB data
  - Adds InterPro domain annotations
- **Output**: Dataset enriched with structural and domain information

### 006-GO-Reasoning-Data-2.ipynb
- **Purpose**: Fixes and refines InterPro data integration
- **Function**: Fetches UniProt dates and corrects issues from previous InterPro processing
- **Output**: Finalized metadata enrichment

---

## Phase 2: Training Data Generation

### 007-reasoning_data.ipynb
- **Purpose**: Creates instruction-tuning dataset
- **Function**: 
  - Generates exact prompts/instructions for model training
  - Formats data for supervised fine-tuning (SFT)
- **Output**: Instruction-formatted training data

### 008-remove-GO-def_add-InterPro.ipynb
- **Purpose**: Testing of training code variations
- **Function**: 
  - Experiments with training without GO definitions
  - Validates InterPro data integration in training pipeline
- **Output**: Validated training configurations

### 009-GOSFT_QA.ipynb
- **Purpose**: Generates GO ontology Q&A dataset for supervised fine-tuning
- **Function**: 
  - Parses GO OBO file and builds tree structures
  - Generates 7 types of Q&A pairs per GO term (category, name, definition, parents, children, siblings, paths)
- **Output**: Comprehensive GO knowledge Q&A dataset (~300K pairs)

### 010-GOSFT_Generation_Refined.ipynb
- **Purpose**: Formats data into natural language SFT format
- **Function**: 
  - Converts reasoning data to QA (Question-Answer) pairs
  - Prepares final training-ready format
- **Output**: Natural language formatted training examples

### 011-go-graph_traversal.ipynb
- **Purpose**: Implements GO graph traversal logic
- **Function**: 
  - Code for traversing GO hierarchy during training/generation
  - Handles GO term relationships and hierarchies
- **Output**: GO graph traversal utilities

---

## Phase 3: Test Set Creation

### 012-test-data_extraction.ipynb
- **Purpose**: Creates CAFA5 test set
- **Function**: 
  - Implements LK/NK (Leave-K/New-K) filtering
  - Extracts and processes test data from GOA
- **Output**: Filtered test dataset

---

## Phase 4: SwissProt Dataset Processing

### 013-swissprot-data.ipynb
- **Purpose**: Processes entire SwissProt dataset release
- **Function**: 
  - Parses SwissProt XML data
  - Converts SwissProt data to reasoning format
- **Output**: SwissProt reasoning dataset

### 014-swissprot-data-2.ipynb
- **Purpose**: Adds more metadata to SwissProt reasoning dataset
- **Function**: Further enrichment of SwissProt data with BLAST and STRING info
- **Output**: Enhanced SwissProt dataset

### 015-swissprot_string_interactions.ipynb
- **Purpose**: Processes STRING database for SwissProt proteins
- **Function**: 
  - Adds PPI data to SwissProt dataset
  - Similar to `cafa5_string_interactions` but for SwissProt
- **Output**: SwissProt dataset with PPI data

### 016-Cafa5ExtendedTestSet_Final.ipynb
- **Purpose**: Constructs temporal holdout test sets from UniProt GOA files
- **Function**: 
  - Downloads and filters GAF files for specified time periods
  - Applies NK/LK filtering and GO hierarchy propagation
- **Output**: Temporal test set with CAFA5 compatibility

---

## Phase 5: Test Set Processing

### 017-Interlabel_Test_Set_Processing.ipynb
- **Purpose**: Creates InterLabel test set
- **Function**: 
  - Adds all metadata and rich data sources
  - Processes InterLabelGO benchmark data
- **Output**: Complete InterLabel test set

### 018-temporal-holdout-ext.ipynb
- **Purpose**: Processes temporal holdout list from UniProt (2022-2025)
- **Function**: 
  - Extracts rich metadata for temporal holdout proteins
  - Creates extended temporal holdout dataset
- **Output**: Temporal holdout dataset with metadata

### 019-temporal-holdout-ext-string.ipynb
- **Purpose**: Gets STRING data for temporal holdout proteins
- **Function**: 
  - Adds PPI information to temporal holdout set
  - Completes temporal holdout dataset
- **Output**: Temporal holdout dataset with PPI data

---

## Phase 6: Additional Processing & Benchmarking

### 020-Interpro_API.ipynb
- **Purpose**: Runs InterProScan for missing InterPro IDs
- **Function**: 
  - Fills gaps in InterPro annotations for test set proteins
  - API-based InterPro annotation retrieval
- **Output**: Complete InterPro annotations

### 021-Benchmarking.ipynb
- **Purpose**: Retrains InterLabel model and benchmarks against CAFA5 SOTA models
- **Function**: 
  - Retrains InterLabelGO+ model
  - Evaluates and compares against state-of-the-art models (NetGO4, SUPERMAGOv2)
- **Output**: Benchmark results and retrained models

---

## Summary Statistics

- **Total notebooks**: 21
- **Main workflow**: CAFA5 dataset creation → Training data generation → Test set creation → SwissProt processing → Benchmarking
- **Key datasets created**: 
  - `cafa5_reasoning` (HuggingFace)
  - SwissProt reasoning dataset
  - Temporal holdout sets (2022-2025)
  - InterLabel test sets
  - GO SFT Q&A dataset
- **Data sources integrated**: 
  - STRING (Protein-Protein Interactions)
  - InterPro (Protein domains)
  - AlphaFold/PDB (Protein structures)
  - BLAST (Sequence similarity)
  - GO annotations (Gene Ontology)
  - UniProt/SwissProt (Protein metadata)

---

## Directory Structure

```
final_data/
├── README.md
├── 001-GO-Dataset.ipynb
├── 002-cafa5_string_interactions.ipynb
├── 003-cafa5_ppi_protein_name_cleaning.ipynb
├── 004-GO-Reasoning-Data.ipynb
├── 005-Protein_Structure_and_Domains.ipynb
├── 006-GO-Reasoning-Data-2.ipynb
├── 007-reasoning_data.ipynb
├── 008-remove-GO-def_add-InterPro.ipynb
├── 009-GOSFT_QA.ipynb
├── 010-GOSFT_Generation_Refined.ipynb
├── 011-go-graph_traversal.ipynb
├── 012-test-data_extraction.ipynb
├── 013-swissprot-data.ipynb
├── 014-swissprot-data-2.ipynb
├── 015-swissprot_string_interactions.ipynb
├── 016-Cafa5ExtendedTestSet_Final.ipynb
├── 017-Interlabel_Test_Set_Processing.ipynb
├── 018-temporal-holdout-ext.ipynb
├── 019-temporal-holdout-ext-string.ipynb
├── 020-Interpro_API.ipynb
└── 021-Benchmarking.ipynb
```

---
