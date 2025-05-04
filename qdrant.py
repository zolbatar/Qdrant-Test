import os
import json
import time
from collections import Counter
from tqdm import tqdm
import numpy as np
import uuid
from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from qdrant_client.http.models import NamedVector, NamedSparseVector

# Data was downloaded from https://nijianmo.github.io/amazon/index.html#complete-data
# Specifically, https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Automotive_5.json.gz

model = SentenceTransformer('all-MiniLM-L6-v2')
vectorizer = TfidfVectorizer(max_features=5000)  # limit features to manageable size

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
    batch_size = 64
    max_retries = 5
    backoff_base = 2  # seconds, doubled on each retry
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
            _upload_with_retry(client, collection_name, points, max_retries, backoff_base)
            points = []  # Reset batch

    # Upload any remaining points
    if points:
        _upload_with_retry(client, collection_name, points, max_retries, backoff_base)

    print(f"Finished inserting {len(all_embeddings)} points.")

def _upload_with_retry(client, collection_name, points, max_retries, backoff_base):
    attempt = 0
    while attempt <= max_retries:
        try:
            client.upsert(
                collection_name=collection_name,
                points=points
            )
            return  # Success, exit the retry loop
        except Exception as e:
            attempt += 1
            if attempt > max_retries:
                print(f"[ERROR] Failed to upload batch after {max_retries} retries. Raising exception.")
                raise e
            else:
                sleep_time = backoff_base * (2 ** (attempt - 1))  # Exponential backoff
                print(f"[WARN] Upload failed on attempt {attempt}. Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)

# Load reviews
amazon_reviews = load_amazon_reviews('Automotive_5.jsonl', limit=1_000_000)

# Work out most popular reviewers
count_reviews_per_user(amazon_reviews)

# Connect to Qdrant
client = QdrantClient(
    url=URL, 
    api_key=API_KEY,
)
check_collection_exists(client)
all_embeddings = create_embeddings()
payloads = [{'user_id': r['user_id'], 'overall': r['overall'], 'text': r['text'], 'summary': r['summary']} for r in amazon_reviews]
assert len(all_embeddings) == len(payloads), "Mismatch between embeddings and payloads"
(sparse_vectors, vocab) = create_sparse_vectors()

# Commented out as already been sent to the database
# send_to_qdrant(payloads, all_embeddings)

query_text = "very bad product"

# Do a query
dense_vector = model.encode(query_text).tolist()
results = client.search(
    collection_name=collection_name,
    query_vector=(dense_vector_name, dense_vector),
    limit=10)
for r in results:
    print(f"Found ID: {r.id}, Score: {r.score}, Payload: {r.payload}")
    
# Results for above:
# Found ID: a961587c-18e7-4636-a92d-bd375299d3ce, Score: 0.94931406, Payload: {'user_id': 'AG9C7PFZAX557', 'overall': 1.0, 'text': 'Really bad product', 'summary': 'One Star'}
# Found ID: 918e1d2c-95b6-4c17-a439-605a99db5c71, Score: 0.8617607, Payload: {'user_id': 'A3EE7AFOSEO7LL', 'overall': 1.0, 'text': 'Bad product.', 'summary': 'Bad'}
# Found ID: e9c81167-a97f-4e7e-9d92-7790bbdd5525, Score: 0.8418441, Payload: {'user_id': 'AO3VUIF03N8JZ', 'overall': 5.0, 'text': 'not bad product', 'summary': 'Five Stars'}
# Found ID: f0589566-b527-469c-83c9-5678915686d5, Score: 0.8084657, Payload: {'user_id': 'A22GG7ADE0GKL7', 'overall': 1.0, 'text': 'Bad bad bad product', 'summary': 'One Star'}
# Found ID: f2c21f59-1a07-4073-88f3-0b0aeb781ffa, Score: 0.78694934, Payload: {'user_id': 'A2KBBC0M3P3GTB', 'overall': 2.0, 'text': 'Bad product!!!', 'summary': 'Two Stars'}
# Found ID: 57479fb1-e687-4f6b-ba0b-4128d0d56bc8, Score: 0.77108353, Payload: {'user_id': 'A2TV6MJKCVW02K', 'overall': 1.0, 'text': 'not a good product', 'summary': 'One Star'}
# Found ID: fdfb6952-72f7-4027-b63c-c83956aae010, Score: 0.7356569, Payload: {'user_id': 'A3316MBPRH8YK7', 'overall': 2.0, 'text': 'Not a very good product', 'summary': 'Two Stars'}
# Found ID: 0f6c58be-a342-4d3c-8f99-116cc9ec2a8a, Score: 0.72714734, Payload: {'user_id': 'A3LII1IHCOPZEP', 'overall': 1.0, 'text': 'Poor product', 'summary': 'One Star'}
# Found ID: 64aaf5b8-5fa4-417e-972e-43df819632a1, Score: 0.72714734, Payload: {'user_id': 'A842YV8YLJ5H', 'overall': 1.0, 'text': 'Poor product', 'summary': 'One Star'}
# Found ID: e1c1f0fe-927a-41a6-b816-662009c874ed, Score: 0.71085507, Payload: {'user_id': 'A2OSTL97ZTXR8L', 'overall': 1.0, 'text': 'Terrible product.  Very difficult to use.', 'summary': 'D

# Do a hybrid search
corpus = [item['summary'] + ' ' + item['text'] for item in amazon_reviews]
vectorizer.fit(corpus)
sparse_matrix = vectorizer.transform(["very bad product"]).tocoo()
sparse_vector = {
    "indices": sparse_matrix.col.tolist(),
    "values": sparse_matrix.data.tolist()
}
results = client.query_points(
    collection_name=collection_name,
    prefetch=[
        models.Prefetch(
            query=models.SparseVector(indices=sparse_matrix.col.tolist(), values=sparse_matrix.data.tolist()),
            using=sparse_vector_name,
            limit=50,
        ),
        models.Prefetch(
            query=dense_vector,
            using=dense_vector_name,
            limit=50,
        ),
    ],
    query=models.FusionQuery(fusion=models.Fusion.RRF),
)
for r in results.points:
    print(f"Found ID: {r.id}, Score: {r.score}, Payload: {r.payload}")

# Results for above
"""
Found ID: a961587c-18e7-4636-a92d-bd375299d3ce, Score: 0.94931406, Payload: {'user_id': 'AG9C7PFZAX557', 'overall': 1.0, 'text': 'Really bad product', 'summary': 'One Star'}
Found ID: 918e1d2c-95b6-4c17-a439-605a99db5c71, Score: 0.8617607, Payload: {'user_id': 'A3EE7AFOSEO7LL', 'overall': 1.0, 'text': 'Bad product.', 'summary': 'Bad'}
Found ID: e9c81167-a97f-4e7e-9d92-7790bbdd5525, Score: 0.8418441, Payload: {'user_id': 'AO3VUIF03N8JZ', 'overall': 5.0, 'text': 'not bad product', 'summary': 'Five Stars'}
Found ID: f0589566-b527-469c-83c9-5678915686d5, Score: 0.8084657, Payload: {'user_id': 'A22GG7ADE0GKL7', 'overall': 1.0, 'text': 'Bad bad bad product', 'summary': 'One Star'}
Found ID: f2c21f59-1a07-4073-88f3-0b0aeb781ffa, Score: 0.78694934, Payload: {'user_id': 'A2KBBC0M3P3GTB', 'overall': 2.0, 'text': 'Bad product!!!', 'summary': 'Two Stars'}
Found ID: 57479fb1-e687-4f6b-ba0b-4128d0d56bc8, Score: 0.77108353, Payload: {'user_id': 'A2TV6MJKCVW02K', 'overall': 1.0, 'text': 'not a good product', 'summary': 'One Star'}
Found ID: fdfb6952-72f7-4027-b63c-c83956aae010, Score: 0.7356569, Payload: {'user_id': 'A3316MBPRH8YK7', 'overall': 2.0, 'text': 'Not a very good product', 'summary': 'Two Stars'}
Found ID: 0f6c58be-a342-4d3c-8f99-116cc9ec2a8a, Score: 0.72714734, Payload: {'user_id': 'A3LII1IHCOPZEP', 'overall': 1.0, 'text': 'Poor product', 'summary': 'One Star'}
Found ID: 64aaf5b8-5fa4-417e-972e-43df819632a1, Score: 0.72714734, Payload: {'user_id': 'A842YV8YLJ5H', 'overall': 1.0, 'text': 'Poor product', 'summary': 'One Star'}
Found ID: e1c1f0fe-927a-41a6-b816-662009c874ed, Score: 0.71085507, Payload: {'user_id': 'A2OSTL97ZTXR8L', 'overall': 1.0, 'text': 'Terrible product.  Very difficult to use.', 'summary': 'Do not buy'}
Found ID: 918e1d2c-95b6-4c17-a439-605a99db5c71, Score: 0.5833334, Payload: {'user_id': 'A3EE7AFOSEO7LL', 'overall': 1.0, 'text': 'Bad product.', 'summary': 'Bad'}
Found ID: 693964a9-1ef1-45ff-b655-9015c4354550, Score: 0.5, Payload: {'user_id': 'A3V1IYRRKXCDZC', 'overall': 2.0, 'text': 'Bad  product.', 'summary': 'Bad product.'}
Found ID: a961587c-18e7-4636-a92d-bd375299d3ce, Score: 0.5, Payload: {'user_id': 'AG9C7PFZAX557', 'overall': 1.0, 'text': 'Really bad product', 'summary': 'One Star'}
Found ID: 8555990b-91ad-4f03-9fc5-9b1fef1207c6, Score: 0.4047619, Payload: {'user_id': 'A2U2O76Y2HJ38R', 'overall': 5.0, 'text': 'not a bad product ,knot a bad product', 'summary': 'not a bad product, knot a bad product'}
Found ID: f0589566-b527-469c-83c9-5678915686d5, Score: 0.4, Payload: {'user_id': 'A22GG7ADE0GKL7', 'overall': 1.0, 'text': 'Bad bad bad product', 'summary': 'One Star'}
Found ID: e9c81167-a97f-4e7e-9d92-7790bbdd5525, Score: 0.375, Payload: {'user_id': 'AO3VUIF03N8JZ', 'overall': 5.0, 'text': 'not bad product', 'summary': 'Five Stars'}
Found ID: f2c21f59-1a07-4073-88f3-0b0aeb781ffa, Score: 0.2777778, Payload: {'user_id': 'A2KBBC0M3P3GTB', 'overall': 2.0, 'text': 'Bad product!!!', 'summary': 'Two Stars'}
Found ID: 40769fd9-0080-4c5a-99b1-8963da8aab94, Score: 0.16666667, Payload: {'user_id': 'A1EN4SYHOFMZK7', 'overall': 1.0, 'text': 'does not work, very bad quality', 'summary': 'very bad'}
Found ID: 76926f36-8780-49ee-b2c9-4d26aabdf4d7, Score: 0.14285715, Payload: {'user_id': 'A3HM85PQFSEDUO', 'overall': 1.0, 'text': 'BAD', 'summary': 'BAD'}
Found ID: 57479fb1-e687-4f6b-ba0b-4128d0d56bc8, Score: 0.14285715, Payload: {'user_id': 'A2TV6MJKCVW02K', 'overall': 1.0, 'text': 'not a good product', 'summary': 'One Star'}
"""