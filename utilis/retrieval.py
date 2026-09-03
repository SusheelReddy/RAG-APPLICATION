from utilis.chunking import chunk_text
from utilis.embedding import get_embeddings

import pickle
import faiss
import os
import numpy as np


def load_faiss_index():

    index_path = "faiss_store/index.faiss"
    mapping_path = "faiss_store/chunk_mapping.pkl"

    # Check whether both files exist and are not empty
    valid = (
        os.path.exists(index_path)
        and os.path.getsize(index_path) > 0
        and os.path.exists(mapping_path)
        and os.path.getsize(mapping_path) > 0
    )

    # -----------------------------------
    # Load existing FAISS index
    # -----------------------------------

    if valid:

        try:

            print("Loading existing FAISS index...")

            index = faiss.read_index(index_path)

            with open(mapping_path, "rb") as f:
                chunk_mapping = pickle.load(f)

            print("FAISS index loaded successfully")
            print("Number of vectors:", index.ntotal)
            print("Number of chunks:", len(chunk_mapping))

            return index, chunk_mapping

        except Exception as e:

            print(
                "Corrupted index detected. Rebuilding...",
                e
            )

    # -----------------------------------
    # Build new FAISS index
    # -----------------------------------

    print("Generating new FAISS index from founder_story.txt...")

    with open(
        "data/founder_story.txt",
        "r",
        encoding="utf-8"
    ) as f:

        text = f.read()

    chunks = chunk_text(text)

    print("Number of chunks:", len(chunks))

    # Get first embedding
    first_embedding = get_embeddings(chunks[0])

    first_embedding = np.array(
        first_embedding,
        dtype="float32"
    )

    print(
        "Embedding dimension:",
        first_embedding.shape
    )

    # Create FAISS index using embedding dimension
    dimension = first_embedding.shape[0]

    index = faiss.IndexFlatL2(dimension)

    chunk_mapping = []

    # Add first embedding
    index.add(
        first_embedding.reshape(1, -1)
    )

    chunk_mapping.append(chunks[0])

    # Add remaining embeddings
    for chunk in chunks[1:]:

        emb = get_embeddings(chunk)

        emb = np.array(
            emb,
            dtype="float32"
        )

        index.add(
            emb.reshape(1, -1)
        )

        chunk_mapping.append(chunk)

    print("FAISS vectors:", index.ntotal)
    print("Chunk mapping:", len(chunk_mapping))

    # -----------------------------------
    # Save FAISS index
    # -----------------------------------

    os.makedirs(
        "faiss_store",
        exist_ok=True
    )

    faiss.write_index(
        index,
        index_path
    )

    with open(
        mapping_path,
        "wb"
    ) as f:

        pickle.dump(
            chunk_mapping,
            f
        )

    print("FAISS index saved successfully")

    return index, chunk_mapping


def retrieve_chunks(
    query,
    index,
    chunk_mapping,
    k=3
):

    query_embedding = get_embeddings(query)

    query_embedding = np.array(
        query_embedding,
        dtype="float32"
    )

    distance, indices = index.search(
        query_embedding.reshape(1, -1),
        k
    )

    return [
        chunk_mapping[i]
        for i in indices[0]
    ]

def retrieve_chunks(query,index,chunk_mapping,k=3):
    query_embedding=get_embeddings(query)
    distance, indices = index.search(np.array([query_embedding]).astype("float32"),k)
    return [chunk_mapping[i] for i in indices[0]]




