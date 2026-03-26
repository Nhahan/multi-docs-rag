# multi-docs-rag

Local multi-document RAG project.

## Purpose

This project ingests local PDF documents, retrieves evidence across multiple documents, and generates grounded answers using only local models.

The goal is simple:

- keep the pipeline local
- keep retrieval and answer generation generic
- answer only from retrieved evidence
- handle cross-document questions without corpus-specific branching

## Current quality focus

The main quality gate is a hard multi-document question that checks:

- paper-side numeric grounding
- cross-document grounding
- refusal only when evidence is actually insufficient

Quality is judged by manual semantic review, not automatic scoring.
