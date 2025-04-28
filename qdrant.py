import os
import json
from collections import Counter
from tqdm import tqdm
import numpy as np
import uuid
from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix

# Data was downloaded from https://nijianmo.github.io/amazon/index.html#complete-data
# Specifically, https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Automotive_5.json.gz

collection_name = "amazon_reviews"
dense_vector_name = "dense"
sparse_vector_name = "sparse"
embedding_file = 'embeddings.npy'
sparse_file  = 'sparse.json'
vocab_file = 'vocab.json'

def load_amazon_reviews(path, limit=None):
    texts = []
    with open(path, 'r') as f:
        for idx, line in enumerate(f):
            if limit and idx >= limit:
                break
            data = json.loads(line)
            texts.append({
                'text': data.get('reviewText', ''),
                'summary': data.get('summary', ''),
                'user_id': data.get('reviewerID', ''),
                'overall': data.get('overall', 0)
            })
    return texts

def count_reviews_per_user(texts):
    user_counter = Counter()
    for item in texts:
        user_id = item.get('user_id')
        if user_id:
            user_counter[user_id] += 1

    print(f"Total unique users: {len(user_counter)}")

    print("Top 10 users with most reviews:")
    for user, count in user_counter.most_common(10):
        print(f"User: {user} - Reviews: {count}")

    # Find users with at least N reviews
    N = 1000
    heavy_users = [user for user, count in user_counter.items() if count >= N]
    print(f"Users with at least {N} reviews: {len(heavy_users)}")

    return user_counter

    """ Output from above
    Total unique users: 186335
    Top 10 users with most reviews:
    User: A2QR9IXLMIDL5U - Reviews: 157
    User: A24FYZZXCMP44U - Reviews: 124
    User: A2V1J3JT5OOZFO - Reviews: 116
    User: AJCHGS1GND4OA - Reviews: 115
    User: A9EL8GNJCW9S8 - Reviews: 114
    User: ANBTTR2QT4C7 - Reviews: 112
    User: A2TM0K7HUH7SLC - Reviews: 109
    User: A365PBEOWM7EI7 - Reviews: 99
    User: A1AKW788238PWQ - Reviews: 96
    User: APKBGB3JBWL5X - Reviews: 95
    Users with at least 1000 reviews: 0
    """

# Check if collection exists
def check_collection_exists(client):
    if not client.collection_exists(collection_name=collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                dense_vector_name: VectorParams(size=384, 
                                                distance=Distance.COSINE, 
                                                quantization_config=models.BinaryQuantization(
                                                    binary=models.BinaryQuantizationConfig(
                                                    always_ram=True))),
            },
            sparse_vectors_config={sparse_vector_name: models.SparseVectorParams()},
            replication_factor=2,
            shard_number=3,
        )

def create_embeddings():
    # Load sentence-transformers model for dense vectors
    model = SentenceTransformer('all-MiniLM-L6-v2')

    all_embeddings = []
    if os.path.exists(embedding_file):
        print(f"Loading embeddings from {embedding_file}...")
        all_embeddings = np.load(embedding_file)
    else:
        print("Embeddings file not found. Creating embeddings...")
        batch_size = 512
        for i in tqdm(range(0, len(amazon_reviews), batch_size)):
            batch = amazon_reviews[i:i+batch_size]
            embeddings = model.encode(batch, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True)
            all_embeddings.append(embeddings)

        # Combine all batches into one big array
        all_embeddings = np.vstack(all_embeddings)

        # Save to disk
        np.save(embedding_file, all_embeddings)
        
    return all_embeddings

# Sparse vectors
def create_sparse_vectors():
    if os.path.exists(sparse_file):
        with open(sparse_file, 'r') as f:
            sparse_vectors = json.load(f)
        with open(vocab_file, 'r') as f:
            tfidf_vocabulary = json.load(f)
        return (sparse_vectors, tfidf_vocabulary)
    else:
        corpus = [item['summary'] + ' ' + item['text'] for item in amazon_reviews]
        vectorizer = TfidfVectorizer(max_features=5000)  # limit features to manageable size
        tfidf_matrix = vectorizer.fit_transform(corpus)

        # Save vocab (as it may change every time)
        fixed_vocab = {str(k): int(v) for k, v in vectorizer.vocabulary_.items()}
        with open(vocab_file, 'w') as f:
            json.dump(fixed_vocab, f)
        
        # Assume your matrix is called tfidf_matrix
        assert isinstance(tfidf_matrix, csr_matrix)

        # List of sparse vectors as dictionaries
        sparse_vectors = []

        for i in range(tfidf_matrix.shape[0]):
            row = tfidf_matrix[i]
            sparse_vector = {int(idx): float(val) for idx, val in zip(row.indices, row.data)}
            sparse_vectors.append(sparse_vector)

        with open(sparse_file, 'w') as f:
            json.dump(sparse_vectors, f)    

        print(f"Converted {len(sparse_vectors)} sparse vectors")
        return (sparse_vectors, fixed_vocab)

# Upsert in batches
def send_to_qdrant(payloads, all_embeddings):
    batch_size = 256
    points = []
    for idx in range(len(all_embeddings)):
        if idx % 1000 == 0:
            print(f"Processing idx={idx}")

        # Convert sparse vector
        sparse = models.SparseVector(
            indices=[int(k) for k in sparse_vectors[idx].keys()],
            values=[float(v) for v in sparse_vectors[idx].values()]
        )

        # Build point structure
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector={
                dense_vector_name: all_embeddings[idx].tolist(),
                sparse_vector_name: sparse,
            },
            payload=payloads[idx]
        )

        points.append(point)

        # Upload batch every `batch_size` points
        if len(points) == batch_size:
            client.upsert(
                collection_name=collection_name,
                points=points
            )
            points = []  # Reset batch

    # Upload any remaining points
    if points:
        client.upsert(
            collection_name=collection_name,
            points=points
        )

    print(f"Finished inserting {len(all_embeddings)} points.")

# Load reviews
amazon_reviews = load_amazon_reviews('Automotive_5.jsonl', limit=1_000_000)

# Work out most popular reviewers
count_reviews_per_user(amazon_reviews)

# Connect to Qdrant
client = QdrantClient(
    url="", 
    api_key="",
)
check_collection_exists(client)
all_embeddings = create_embeddings()
payloads = [{'user_id': r['user_id'], 'overall': r['overall'], 'text': r['text'], 'summary': r['summary']} for r in amazon_reviews]
assert len(all_embeddings) == len(payloads), "Mismatch between embeddings and payloads"
(sparse_vectors, vocab) = create_sparse_vectors()
send_to_qdrant(payloads, all_embeddings)
