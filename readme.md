# Soba Inferer

This project consists of 2 segments:
- Soba Compressor / Embedder
- Soba Inferer

## Soba Compressor / Embedder

`Soba Compressor` supports the reading and sentence splitting of `*.pdf` and `*.txt` files provided within a directory. It performs chunking of the retrieved data based on `sentence + word count` and generates a `compressed_*.txt` file in the `soba-inferer/compressed/*` directory.

The compressed files are then subsequently passed to the `Soba Embedder` which performs embedding to convert it to a vector database for subsequent reference in a KNN search.

Refer to the concept: [Hugging Face - Advanced Rag](https://huggingface.co/learn/cookbook/advanced_rag)

## Soba Inferer

This will serve as the LLM Inferer that will serve as the endpoint for the LLM Server.

## Installations

```python
# You may be required to perform an installation within the code using the "nltk" library
# before being able to run the tcp_core.py file (inferer)
import nltk

if __name__== "__main__":
    nltk.download('punkt')
```

## How To Use

### Dependency Installation

```cmd
pip install -r requirements.txt
```

### Standalone Data Compression

```python
from custom_compressor import GenerateCompressedFiles

"""
Reads all *.pdf and *.txt files from "your_file_directory" and
writes them to "soba-inferer/compressed/*" where * represents
the read file's prefix directory
"""
if __name__== "__main__":
    GenerateCompressedFiles("your_file_directory")
```

### Standalone Data Embedding

```python
from custom_embedder import GenerateAllEmbeddings

"""
Creates a single embedding vector database using the 'all-MiniLM-L6-v2'
model from SentenceTransformer after reading each "compressed_file" from
"your_compressed_file_directory"
"""
if __name__== "__main__":
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = GenerateAllEmbeddings("your_compressed_file_directory", model)
```

### Run Inferer

```bash
# Run the following command in a terminal
# 192.168.0.1 - Refers to the IPV4 address that the inferer listens to
# 7060        - Refers to the Port that the inferer listens to
# 11434       - Refers to the Port that the ollama client listens to
#               - Could be in Docker Container / Localhost
python tcp_core.py 192.168.0.1 7060 11434
```