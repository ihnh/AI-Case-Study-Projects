"""
Embeddings with Amazon Bedrock
--------------------------------------------------------------
Prerequisites:
    pip install boto3 numpy matplotlib scikit-learn

AWS credentials must be configured:
    aws configure   OR   set environment variables:
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import boto3
from sklearn.manifold import TSNE


# ──────────────────────────────────────────────
# Generate Embeddings with Amazon Bedrock
# ──────────────────────────────────────────────

# Initialize the Bedrock client
# Make sure AWS region supports Bedrock (e.g. us-east-1)
bedrock = boto3.client(
    service_name="bedrock-runtime",   # NOTE: use 'bedrock-runtime'
    region_name="us-east-1"
)


def get_embedding(text: str) -> list[float]:
    """
    Call Amazon Titan Embed Text v1 to get a vector embedding for a string.
    Returns a list of floats (1536 dimensions for Titan Embed v1).
    """
    body = json.dumps({"inputText": text})
    response = bedrock.invoke_model(
        modelId="amazon.titan-embed-text-v1",
        body=body,
        contentType="application/json",
        accept="application/json"
    )
    response_body = json.loads(response["body"].read())
    return response_body["embedding"]


# --- Test it ---
print("=" * 60)
print("PART 2: Generating Embeddings")
print("=" * 60)

test_text = "Embeddings are useful for natural language processing tasks."
test_embedding = get_embedding(test_text)

print(f"Input text   : {test_text}")
print(f"Embedding dim: {len(test_embedding)}")        # Titan v1 → 1536 dimensions
print(f"First 5 vals : {test_embedding[:5]}\n")

# Generate embeddings for the 4 exercise texts
texts = [
    "The cat sat on the mat.",
    "A feline rested on a rug.",
    "Dogs are loyal companions.",
    "Artificial intelligence is reshaping technology.",
]

print("Generating embeddings for 4 sample texts...")
embeddings = [get_embedding(t) for t in texts]
print("Done!\n")

# Quick sanity check — print first 3 values of each
for text, emb in zip(texts, embeddings):
    print(f"  '{text[:40]}...' → [{emb[0]:.4f}, {emb[1]:.4f}, {emb[2]:.4f}, ...]")


# ──────────────────────────────────────────────
# Visualize Embeddings in 2D (t-SNE)
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("PART 3: Visualizing Embeddings with t-SNE")
print("=" * 60)

# t-SNE reduces 1536-D vectors down to 2-D for plotting.
# perplexity must be < n_samples; with 4 samples we use 2.
tsne = TSNE(n_components=2, random_state=42, perplexity=2)
reduced = tsne.fit_transform(np.array(embeddings))

# Plot
plt.figure(figsize=(10, 7))
colors = ["#e74c3c", "#e67e22", "#2ecc71", "#3498db"]

for i, (text, color) in enumerate(zip(texts, colors)):
    plt.scatter(reduced[i, 0], reduced[i, 1], color=color, s=120, zorder=5)
    plt.annotate(
        text,
        (reduced[i, 0], reduced[i, 1]),
        textcoords="offset points",
        xytext=(8, 5),
        fontsize=9,
        color=color,
        fontweight="bold"
    )

plt.title("2D t-SNE Visualization of Text Embeddings", fontsize=14, fontweight="bold")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("embeddings_visualization.png", dpi=150)
plt.show()

print("Plot saved to embeddings_visualization.png")
print("""
Expected observation:
  - 'The cat sat on the mat.' and 'A feline rested on a rug.' should be CLOSE
    (both describe a cat/feline resting on a soft surface).
  - 'Dogs are loyal companions.' is somewhat different (animals but different context).
  - 'Artificial intelligence is reshaping technology.' should be FAR from the rest
    (completely different semantic domain).
""")


# ──────────────────────────────────────────────
# Similarity Search
# ──────────────────────────────────────────────
print("=" * 60)
print("PART 4: Similarity Search with Cosine Similarity")
print("=" * 60)


def cosine_similarity(a: list, b: list) -> float:
    """
    Cosine similarity between two vectors.
    Result ranges from -1 (opposite) to 1 (identical direction).
    In practice for embeddings you'll see values roughly 0.5–1.0.
    """
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def find_most_similar(query_embedding, corpus_embeddings, corpus_texts):
    """
    Compare a query embedding against a list of corpus embeddings.
    Returns (most_similar_text, similarity_score) and prints a ranked list.
    """
    scored = [
        (text, cosine_similarity(query_embedding, emb))
        for text, emb in zip(corpus_texts, corpus_embeddings)
    ]
    scored.sort(key=lambda x: x[1], reverse=True)

    print("  Ranked results:")
    for rank, (text, score) in enumerate(scored, 1):
        print(f"    {rank}. [{score:.4f}] {text}")

    return scored[0]  # (text, score) of the best match


# --- Test queries ---
queries = [
    "A kitty lounged on a carpet.",          # should match cat/feline texts
    "Machine learning is changing the world.", # should match AI text
    "My dog loves to play fetch.",            # should match dogs text
]

for query in queries:
    print(f"\nQuery: \"{query}\"")
    query_emb = get_embedding(query)
    best_text, best_score = find_most_similar(query_emb, embeddings, texts)
    print(f"  → Best match : \"{best_text}\"")
    print(f"  → Similarity : {best_score:.4f}")


# ──────────────────────────────────────────────
# How This Relates to RAG
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("PART 5: Connecting Embeddings to RAG")
print("=" * 60)
print("""
In a real RAG pipeline, the similarity search above IS the retrieval step:

  1. INDEXING (done once, offline)
     ┌─────────────────────────────────────────────────────┐
     │  Raw documents                                       │
     │      ↓  chunk into paragraphs                       │
     │  Chunks  →  get_embedding(chunk)  →  store vectors  │
     │                           (in a vector DB like FAISS│
     │                            or Amazon OpenSearch)     │
     └─────────────────────────────────────────────────────┘

  2. QUERY TIME (for every user question)
     ┌─────────────────────────────────────────────────────┐
     │  User question  →  get_embedding(question)          │
     │      ↓  cosine_similarity search                    │
     │  Top-K most relevant chunks retrieved               │
     │      ↓                                              │
     │  Prompt = "Use this context: {chunks}\\n            │
     │            Answer: {user_question}"                  │
     │      ↓                                              │
     │  LLM (Claude / Titan / etc.) generates final answer │
     └─────────────────────────────────────────────────────┘

The small 4-sentence demo above is a micro version of this.
Swap 'texts' for thousands of document chunks, and
'embeddings' for a proper vector store, and you have production RAG.
""")

print("Complete! ✓")