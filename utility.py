import glob
import numpy as np
from sklearn.neighbors import NearestNeighbors


def KnnSearch(question_embedding, embeddings, k=10):
    X = np.array([item['embedding'] for article in embeddings for item in article['embeddings']])
    source_texts = [item['source'] for article in embeddings for item in article['embeddings']]
    
    # Fit a KNN model on the embeddings
    knn = NearestNeighbors(n_neighbors=k, metric='cosine')
    knn.fit(X)

    # In case we have too few embedded data
    if k > knn.n_samples_fit_:
        k = knn.n_samples_fit_
    
    # Find the indices and distances of the k-nearest neighbors
    distances, indices = knn.kneighbors(question_embedding, n_neighbors=k)
    
    # Get the indices and source texts of the best matches
    best_matches = [(indices[0][i], source_texts[indices[0][i]]) for i in range(k)]
    
    return best_matches

def FindValidFilesInDirectory(directory):
    pkl_file_list = glob.glob(directory + '/**/*.pkl', recursive=True)
    return pkl_file_list