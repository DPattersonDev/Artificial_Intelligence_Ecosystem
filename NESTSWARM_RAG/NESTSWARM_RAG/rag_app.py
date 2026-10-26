from pathlib import Path
import re
import numpy as np
import faiss
import requests

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, CrossEncoder


# ------------------------------------------------------------
# Adjustable project parameters
# ------------------------------------------------------------
chunk_size = 500
chunk_overlap = 50

embedding_model_name = "sentence-transformers/all-distilroberta-v1"
reranker_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"

top_k = 20
top_m = 8

llm_model = "qwen3:4b"
ollama_url = "http://localhost:11434/api/chat"

document_path = Path(__file__).with_name("Selected_Document.txt")


def load_document(path: Path) -> str:
    """Load the selected source document."""
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path.name}. Keep it in the same folder as rag_app.py."
        )

    text = path.read_text(encoding="utf-8").strip()

    if not text:
        raise ValueError(f"{path.name} is empty.")

    return text


def split_document(text: str) -> list[str]:
    """Split the source into overlapping text chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = [chunk.strip() for chunk in splitter.split_text(text)]
    return [chunk for chunk in chunks if chunk]


def build_vector_index(chunks: list[str]):
    """Create embeddings and place them in a local FAISS index."""
    print(f"Loading embedding model: {embedding_model_name}")
    embedding_model = SentenceTransformer(embedding_model_name)

    print("Creating document embeddings...")
    embeddings = embedding_model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return embedding_model, index


def retrieve_candidates(
    question: str,
    chunks: list[str],
    embedding_model: SentenceTransformer,
    index: faiss.Index,
) -> list[str]:
    """Use FAISS to retrieve candidate chunks for a question."""
    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    candidate_count = min(top_k, len(chunks))
    _, indices = index.search(question_embedding, candidate_count)

    candidates = []
    seen = set()

    for idx in indices[0]:
        if idx < 0:
            continue

        chunk = chunks[int(idx)]
        if chunk not in seen:
            candidates.append(chunk)
            seen.add(chunk)

    return candidates


def rerank(question: str, candidates: list[str], reranker: CrossEncoder) -> list[str]:
    """Rerank FAISS candidates with a cross-encoder."""
    if not candidates:
        return []

    pairs = [[question, passage] for passage in candidates]
    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(candidates, scores),
        key=lambda item: float(item[1]),
        reverse=True,
    )

    return [passage for passage, _ in ranked[: min(top_m, len(ranked))]]


def clean_model_response(text: str) -> str:
    """Remove Qwen thinking blocks if Ollama returns them."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def ask_ollama(question: str, context_chunks: list[str]) -> str:
    """Send the reranked context and question to local Qwen through Ollama."""
    if not context_chunks:
        return "I could not retrieve relevant context from the document."

    context = "\n\n---\n\n".join(context_chunks)

    system_prompt = (
        "You are a retrieval-augmented assistant. "
        "Answer only from the supplied context. "
        "Do not use outside knowledge. "
        "If the context does not contain enough information, say that you do not know "
        "based on the provided document. "
        "Give a direct, concise answer and do not include hidden reasoning."
    )

    user_prompt = f"""CONTEXT:
{context}

QUESTION:
{question}

Answer the question using only the context above."""

    payload = {
        "model": llm_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 500,
        },
    }

    try:
        response = requests.post(ollama_url, json=payload, timeout=180)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            "Could not reach Ollama. Make sure Ollama is running and "
            f"that the model '{llm_model}' has been pulled."
        ) from exc

    data = response.json()
    answer = data.get("message", {}).get("content", "").strip()

    if not answer:
        return "Ollama returned an empty response."

    return clean_model_response(answer)


def main():
    print("\nNESTSWARM Local RAG")
    print("=" * 40)
    print(f"chunk_size = {chunk_size}")
    print(f"chunk_overlap = {chunk_overlap}")
    print(f"top_k = {top_k}")
    print(f"top_m = {top_m}")
    print()

    document = load_document(document_path)
    chunks = split_document(document)

    print(f"Loaded {len(chunks)} document chunks.\n")

    embedding_model, index = build_vector_index(chunks)

    print(f"\nLoading reranker: {reranker_model_name}")
    reranker = CrossEncoder(reranker_model_name)

    print("\nReady. Ask a question about NESTSWARM.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        question = input("Question: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        if not question:
            continue

        candidates = retrieve_candidates(
            question,
            chunks,
            embedding_model,
            index,
        )

        reranked_chunks = rerank(question, candidates, reranker)
        answer = ask_ollama(question, reranked_chunks)

        print(f"\nAnswer: {answer}\n")


if __name__ == "__main__":
    main()
