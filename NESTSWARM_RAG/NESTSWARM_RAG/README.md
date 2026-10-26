# NESTSWARM Local RAG Project

## Project Overview

This project builds a local Retrieval-Augmented Generation (RAG) system using a document about my game project, **NESTSWARM**.

The system uses:

- Sentence Transformers for embeddings
- FAISS for local vector search
- A cross-encoder for reranking
- Ollama for local model hosting
- Qwen 3 4B as the local language model
- Python for the full pipeline

The final RAG application runs locally and answers questions using information retrieved from `Selected_Document.txt`.

## Files

- `Selected_Document.txt` — source knowledge document about NESTSWARM
- `text_extractor.py` — utility that can extract text from a web page, PDF, or existing text file
- `rag_app.py` — complete local RAG pipeline
- `requirements.txt` — required Python packages
- `README.md` — project explanation, experiments, deep-dive questions, and reflection

## Selected Document

For this project, I used a text document describing my game project, **NESTSWARM**. The document contains information about the game's factions, characters, combat system, round structure, upgrades, game modes, progression, monetization, and current development priorities.

I selected this document because I already understand the information in it. This makes it easier to judge whether the RAG system retrieves the correct information and whether Qwen generates answers that are grounded in the source.

## RAG Pipeline

1. The program loads `Selected_Document.txt`.
2. `RecursiveCharacterTextSplitter` divides the document into overlapping chunks.
3. `sentence-transformers/all-distilroberta-v1` converts the chunks into vector embeddings.
4. FAISS stores the embeddings in a local vector index.
5. The user's question is embedded and compared with the stored chunks.
6. The top candidate chunks are retrieved from FAISS.
7. `cross-encoder/ms-marco-MiniLM-L-6-v2` reranks the candidate chunks.
8. The highest-ranked passages are sent to `qwen3:4b` through the local Ollama API.
9. Qwen is instructed to answer only from the supplied context.

## Final Parameters

```python
chunk_size = 500
chunk_overlap = 50
embedding_model_name = "sentence-transformers/all-distilroberta-v1"
reranker_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
top_k = 20
top_m = 8
llm_model = "qwen3:4b"
ollama_url = "http://localhost:11434/api/chat"
```

## Installation

Install Ollama and pull the model:

```bash
ollama pull qwen3:4b
```

Create and activate a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the Python packages:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the application:

```bash
python rag_app.py
```

## RAG Testing

I tested the system with three main questions.

### Question 1

**Question:** Which characters belong to the Vesper faction?

**Observed answer:** The system correctly identified Veyra, Zekrin, Melora, Tharos, and Aurex.

### Question 2

**Question:** How do upgrades work between rounds?

**Observed answer:** The system correctly explained that after each round the player receives one upgrade choice that can strengthen an individual unit, improve a broader team strategy, or increase the effectiveness of a synergy.

### Question 3

**Question:** What are the planned game modes for NESTSWARM?

**Observed answer:** The system correctly identified 1v1 and 4v4 as the planned game modes.

### Grounding Test

I also asked:

**Question:** Who is the best character in the game?

The source document did not identify any character as the best. The RAG system correctly stated that the document did not contain enough information to name a best character instead of inventing an unsupported answer. This was a useful test of grounding.

## Chunk Size and Overlap Experiments

I used the same three main questions for each configuration.

### Experiment 1: 500 / 50

- `chunk_size = 500`
- `chunk_overlap = 50`

All three questions were answered correctly.

This configuration gave the best overall balance. It provided enough surrounding context for the answer without pulling in too much unnecessary information.

### Experiment 2: 300 / 50

- `chunk_size = 300`
- `chunk_overlap = 50`

All three questions were still answered correctly.

However, the smaller chunks caused some retrieved information to become more fragmented. This was most noticeable in the upgrade question, where related information was spread across smaller passages.

### Experiment 3: 700 / 100

- `chunk_size = 700`
- `chunk_overlap = 100`

All three questions were again answered correctly.

The larger chunks preserved more surrounding context and provided richer details, especially for the upgrade system and 4v4 mode. However, the responses also became more verbose and sometimes included more information than was necessary to answer the question directly.

### Experiment Comparison

The **500 / 50** configuration worked best for this document.

The **300 / 50** configuration was accurate but sometimes fragmented closely related information.

The **700 / 100** configuration preserved more context but increased the amount of unnecessary information supplied to the model.

For the final application, I restored:

- `chunk_size = 500`
- `chunk_overlap = 50`

## Five Deep-Dive Questions

### 1. What does the embedding model do in this RAG system?

The embedding model converts document chunks and user questions into numerical vectors. These vectors represent semantic meaning, allowing the system to compare the meaning of a question with the meaning of document passages instead of depending only on exact keyword matches.

### 2. What does FAISS do?

FAISS stores the document embeddings and performs fast similarity searches. When the user asks a question, the application embeds the question and uses FAISS to find the chunks that are closest to it in vector space.

### 3. Why is chunk overlap useful?

Chunk overlap repeats part of the text between neighboring chunks. This helps prevent important information from being lost when a sentence or idea falls near a chunk boundary. Too much overlap can also cause duplicate or unnecessary information to be retrieved.

### 4. Why use a cross-encoder after FAISS retrieval?

FAISS is fast and useful for retrieving candidate passages, but the cross-encoder can compare the complete question and passage together more carefully. Reranking improves the chance that the most relevant chunks are sent to the language model.

### 5. Why is the prompt important in a RAG system?

The prompt controls how the language model uses the retrieved passages. In this project, Qwen is instructed to answer only from the supplied context and to admit when the context does not contain enough information. This reduces unsupported answers and keeps the model grounded in the selected document.

## Reflection

This project helped me understand how the individual parts of a RAG system work together. The language model is only the final stage. Before Qwen receives the question, the source document is split into chunks, converted into embeddings, searched with FAISS, and reranked with a cross-encoder.

Using a document about NESTSWARM made testing easier because I already knew what information should and should not be present. When I asked factual questions about the Vesper faction, upgrade system, and game modes, the system retrieved the correct information. When I asked which character was the best, the source did not contain an answer and the system correctly refused to invent one.

The chunking experiment also showed that retrieval settings affect answer quality. Smaller chunks can be more focused, but they can separate information that belongs together. Larger chunks preserve more context, but they can introduce extra information and make responses more verbose. For this document, a chunk size of 500 with an overlap of 50 gave the best balance.

AI assistance was used to help organize the Python pipeline, troubleshoot the local environment, explain RAG concepts, and compare the behavior of different chunk settings. I still tested the application locally with Ollama and recorded the behavior of the system using the actual NESTSWARM source document.
