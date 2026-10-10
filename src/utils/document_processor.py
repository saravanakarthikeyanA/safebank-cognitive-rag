import os

def chunk_text(text: str, chunk_size: int = 200, chunk_overlap: int = 30) -> list[str]:
    """Split text into chunks by word count with overlap."""
    words = text.split()
    chunks = []
    if not words:
        return chunks

    i = 0
    while i < len(words):
        chunk_words = words[i:i + chunk_size]
        chunks.append(" ".join(chunk_words))
        # Ensure we make forward progress
        step = chunk_size - chunk_overlap
        if step <= 0:
            step = 1
        i += step
    return chunks

def load_and_chunk_policy(file_path: str, category: str, chunk_size: int = 200, chunk_overlap: int = 30) -> list[dict]:
    """
    Reads a policy text file and splits it into chunks with metadata.

    Args:
        file_path: Path to the text/policy document.
        category: The banking category (e.g. 'Fraud', 'KYC', 'Loan', 'Account Access').
        chunk_size: Target size of each chunk in words.
        chunk_overlap: Word overlap between chunks.

    Returns:
        list[dict]: List of document chunks with 'text' and 'metadata'.
    """
    if not os.path.exists(file_path):
        print(f"Warning: Policy file {file_path} not found.")
        return []

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    chunks = chunk_text(content, chunk_size, chunk_overlap)
    documents = []
    for idx, chunk in enumerate(chunks):
        documents.append({
            "text": chunk,
            "metadata": {
                "source": os.path.basename(file_path),
                "category": category,
                "chunk_id": f"{os.path.basename(file_path)}_chunk_{idx}"
            }
        })
    return documents
