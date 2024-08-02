import os
import ollama
import numpy as np
import nltk
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from mattsollamatools import chunker

from custom_compressor import FindValidFilesInDirectory

# Perform K-nearest neighbors (KNN) search
def KnnSearch(question_embedding, embeddings, k=5):
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


def GenerateEmbeddings(compressed_filename, model):
    with open(compressed_filename, "r") as input_doc:
        if input_doc.closed:
            print(compressed_filename + ' could not be opened to be embedded')
            return
        
        compressed_text = input_doc.read()
        if compressed_text.length == 0:
            return False, {}
        
        chunks = chunker(compressed_text)
        embeddings = model.encode(chunks)

        story = {}
        story['embeddings'] = []
        for (chunk, embedding) in zip(chunks, embeddings):
            item = {}
            item['source'] = chunk
            item['embedding'] = embedding
            item['sourcelength'] = len(chunk)
            story['embeddings'].append(item)

        return True, story
    

def GenerateAllEmbeddings(folder_directory, model, outfile_directory = ''):
    compressed_files = FindValidFilesInDirectory(folder_directory)

    all_embeddings = []
    for filename in compressed_files:
        status, embeddings = GenerateEmbeddings(filename, model)
        if status == True:
            all_embeddings.append(embeddings)

    if outfile_directory != '':
        with open(outfile_directory, 'wb') as f:
            pickle.dump(all_embeddings, f)

    return all_embeddings


def MakeQuery(user_prompt, all_embeddings, model):
    question_embedding = model.encode([user_prompt])   
    best_matches = KnnSearch(question_embedding, all_embeddings, k=5) # Play with this value, higher seems to result in inaccurate results

    sourcetext = ""
    for i, (index, source_text) in enumerate(best_matches, start=1):
        sourcetext += f"{i}. Index: {index}, Source Text: {source_text}"

    response = ollama.generate(
        model= "phi3:3.8b-mini-4k-instruct-q6_K",
        prompt=user_prompt,
        system=f"Use this information only if you are unable to accurately provide an answer to the question: {sourcetext}",
        stream=False,
    )

    print(response['response'])



if __name__ == "__main__":
    nltk.download('punkt')
    model = SentenceTransformer('all-mpnet-base-v2')
    all_embeddings = GenerateAllEmbeddings(os.path.dirname(__file__) + '\\compressed\\', model)

    while True:
        user_prompt = input('Ask the model a question (or /bye): ')
        if (user_prompt == '/bye'):
            break
        MakeQuery(user_prompt, all_embeddings, model)