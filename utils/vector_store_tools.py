from langchain.embeddings import HuggingFaceEmbeddings
import os

def generate_vector_store(data, 
                          cache_loc, 
                          embedding_model_name='sentence-transformers/all-mpnet-base-v2'):

    # Load HF model that will be used to create the embeddings
    hf = HuggingFaceEmbeddings(model_name=embedding_model_name)
    model_kwargs = {'device': 'cuda'}
    encode_kwargs = {'normalize_embeddings': False}
    hf = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        cache_folder = cache_loc
    )

    # Load the vector store that will be used to store/search the embeddings
    from langchain.vectorstores import FAISS

    # Create embeddings for each eval
    db = FAISS.from_documents(data, hf)
    
    return db
    
    
def load_saved_vector_store(cache_loc, 
                            embedding_model_name='sentence-transformers/all-mpnet-base-v2',
                            local_loc=os.path.join('data','faiss_index'),
                           ):

    # Load HF model that will be used to create the embeddings
    hf = HuggingFaceEmbeddings(model_name=embedding_model_name)
    model_kwargs = {'device': 'cuda'}
    encode_kwargs = {'normalize_embeddings': False}
    hf = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        cache_folder = cache_loc
    )

    # Load the vector store that will be used to store/search the embeddings
    from langchain.vectorstores import FAISS

    # Create embeddings for each eval
    db = FAISS.load_local(local_loc, embeddings=hf)
    
    return db

    
def cluster_docs_using_embeddings(db, 
                                  n_clusters = 3,
                                  n_init = 'auto'):
    from sklearn.cluster import KMeans
    
    # Get raw vectors out for clustering
    raw_vecs = {idx:db.index.reconstruct(idx) for idx in db.index_to_docstore_id.keys()}

    # Extract embeddings from the dictionary
    embeddings = list(raw_vecs.values())

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=n_init).fit(embeddings)

    # Cluster labels for each embedding
    cluster_labels = kmeans.labels_

    # Assigning the cluster labels back to the original dictionary
    for idx, label in zip(raw_vecs.keys(), cluster_labels):
        raw_vecs[idx] = {'embedding': raw_vecs[idx], 'cluster': label}

    # The embeddings_dict now has the embeddings and their respective cluster labels
    clustered_docs = raw_vecs
    return clustered_docs


def get_relevant_docs(query, 
                      db,
                      num_docs=5):
    docs = db.similarity_search(query, k=num_docs)
    return docs


def get_cluster_summaries(data, clustered_docs, llm, n_clusters):
    # Create aggregated documents representing each cluster
    doc_clusters = [[data[i] for i in range(len(data)) if clustered_docs[i]['cluster'] == cluster_idx] for cluster_idx in range(n_clusters)]
    # doc_clusters is now a list of lists, where each inner list contains the docs associated with one of the clusters.

    from langchain.chains.summarize import load_summarize_chain
    
    cluster_summaries = []
    
    for cluster in doc_clusters:
        chain = load_summarize_chain(llm, chain_type="map_reduce")
        cluster_summary = chain.run(cluster)
        cluster_summaries.append(cluster_summary)
    
    return cluster_summaries