"""
Document chunking utilities.
Splits large documents into smaller chunks for embedding.
"""
from typing import List


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Text to split
        chunk_size: Maximum characters per chunk
        chunk_overlap: Overlap between chunks (for context continuity)
    
    Returns:
        List of text chunks
    
    Example:
        "Hello world this is a test" with chunk_size=10, overlap=3
        -> ["Hello worl", "orld this", "his is a ", "s a test"]
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - chunk_overlap
    
    return chunks


def chunk_by_sentences(
    text: str,
    max_chunk_size: int = 500
) -> List[str]:
    """
    Split text by sentences, grouping them into chunks.
    Better quality than character-based chunking.
    """
    # Simple sentence splitting (can be improved with nltk/spacy)
    sentences = []
    current = ""
    
    for char in text:
        current += char
        if char in ".!?" and len(current) > 1:
            sentences.append(current.strip())
            current = ""
    
    if current.strip():
        sentences.append(current.strip())
    
    # Group sentences into chunks
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chunk_size:
            current_chunk += " " + sentence if current_chunk else sentence
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks


def chunk_file(file_path: str, chunk_size: int = 500) -> List[dict]:
    """
    Read and chunk a text file.
    
    Returns:
        List of dicts with 'text' and 'metadata'
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    chunks = chunk_by_sentences(content, chunk_size)
    
    return [
        {
            "text": chunk,
            "metadata": {
                "source": file_path,
                "chunk_index": i
            }
        }
        for i, chunk in enumerate(chunks)
    ]
