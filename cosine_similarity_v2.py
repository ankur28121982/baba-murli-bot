import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity



BASE_DIR = os.path.dirname(__file__)
chunks_file_path = os.path.join(BASE_DIR, "chunks.json")

# Load chunks from the JSON file (replace with your path)
# chunks_file_path = r'C:\Users\Bk Pallavi\Desktop\Baba_bot\output\chunks.json'

# Load chunks from the file
with open(chunks_file_path,"r", encoding="utf-8", errors="replace") as f:
    chunks = json.load(f)

# Initialize the model for embeddings (using 'paraphrase-MiniLM-L6-v2')
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Generate embeddings for each chunk
chunk_embeddings = []
for chunk in chunks:
    chunk_embeddings.append(model.encode(chunk['content']))  # 'content' is the key for chunk text

# Convert to numpy array
chunk_embeddings = np.array(chunk_embeddings)

# Function to search for the most relevant chunk based on cosine similarity
def search(query, chunks, chunk_embeddings, top_n=8):
    # Encode the query to get its vector representation
    query_embedding = model.encode(query).reshape(1, -1)

    # Calculate cosine similarity between the query and all chunk embeddings
    similarity_scores = cosine_similarity(query_embedding, chunk_embeddings)
    
    # Get the indices of the top N most similar chunks
    top_n_indices = np.argsort(similarity_scores[0])[::-1][:top_n]

    # Select the top N chunks
    top_n_chunks = [chunks[i] for i in top_n_indices]
    
    # Re-rank the top N chunks to select the best one (based on highest similarity score)
    re_ranked_chunk = top_n_chunks[0]  # Since top_n_chunks are sorted by similarity score

    # Return the re-ranked chunk and its similarity score
    return re_ranked_chunk, similarity_scores[0][top_n_indices[0]], top_n_chunks

# Example query
query = "why baba calling all of us madhuban residents?"

# Perform search
result, score, top_chunks = search(query, chunks, chunk_embeddings)

# Output the result
print(f"Best chunk:\n{result['content']}\nScore: {score}")

# Optionally, print the top 20 chunks (for reference)
print("\nTop 20 Relevant Chunks (for reference):")
for idx, chunk in enumerate(top_chunks):
    print(f"Rank {idx+1}: Score: {cosine_similarity(model.encode(query).reshape(1, -1), model.encode(chunk['content']).reshape(1, -1))[0][0]} - {chunk['content'][:100]}...")  # Print first 100 characters for preview
