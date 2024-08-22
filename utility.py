import glob
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import util


def KnnSearch(question_embedding, embeddings, k=10):
    X = np.array([item['embedding'] for article in embeddings for item in article['embeddings']])
    source_texts = [item['source'] for article in embeddings for item in article['embeddings']]
    text_source_location = [item['docname'] for article in embeddings for item in article['embeddings']]
    
    # Fit a KNN model on the embeddings
    knn = NearestNeighbors(n_neighbors=k, metric='cosine')
    knn.fit(X)

    # In case we have too few embedded data
    if k > knn.n_samples_fit_:
        k = knn.n_samples_fit_
    
    # Find the indices and distances of the k-nearest neighbors
    distances, indices = knn.kneighbors(question_embedding, n_neighbors=k)
    
    # Get the indices and source texts of the best matches
    best_matches = [( indices[0][i],
                      source_texts[indices[0][i]],
                      text_source_location[indices[0][i]] ) for i in range(k)]
    
    return best_matches

def SemanticSearch(query_embeddings, database, k):
    data_embeddings = np.array([item['embedding'] for article in database for item in article['embeddings']])
    source_texts = [item['source'] for article in database for item in article['embeddings']]
    text_source_location = [item['docname'] for article in database for item in article['embeddings']]

    hits = util.semantic_search(query_embeddings, data_embeddings, score_function=util.cos_sim, top_k=k)

    generated_result =  [ (source_texts[neighbours['corpus_id']], text_source_location[neighbours['corpus_id']]) for neighbours in hits[0] ]
    return generated_result

def FindValidFilesInDirectory(directory):
    pkl_file_list = glob.glob(directory + '/**/*.pkl', recursive=True)
    return pkl_file_list