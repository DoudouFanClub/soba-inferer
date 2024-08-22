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
    # Use 'all-MiniLM-L6-v2' if you are looking for embedding speed
    # Use 'all-mpnet-base-v2' if you are looking for quality
    model = SentenceTransformer('all-mpnet-base-v2')
    embeddings = GenerateAllEmbeddings("your_compressed_file_directory", model)
```

For more Sentence Transformer Information - Refer to [Sentence Transformer Pretrained Models](https://www.sbert.net/docs/sentence_transformer/pretrained_models.html#semantic-search-models)

### Run Inferer

```bash
# Run the following command in a terminal
# 192.168.0.1 - Refers to the IPV4 address that the inferer listens to
# 7060        - Refers to the Port that the inferer listens to
# 11434       - Refers to the Port that the ollama client listens to
#               - Could be in Docker Container / Localhost
python tcp_core.py 192.168.0.1 7060 11434
```

## Preparation of Ollama Dependencies

### Docker Containerization of Ollama Models

```bash
# Creating a Docker Container from a Docker Image
# You are only required to run this command once in order to
# construct a Docker Container
docker create --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama_1 ollama/ollama
```

The way this works is as follows:
- `--gpus=all` - Allows for access to all GPUs known to the OS. Applicable customizations are `--gpus=1` for example where we use the GPU located in Slot 1 (Refer to the Task Manager to see which GPU is located in the slot)
- `-v ollama:/root/.ollama` - Creates a Docker Volume that may be shared across all Ollama instances. `ollama` refers to the folder created on the `Host Machine` and `/root/.ollama` refers to the folder created in the `Docker Container`.
- `-p 11434:11434` - Specifies a port `11434 (left)` that is accessible on the `Host Machine` and `11434 (right)` that connects to the port `11434` on the `Host Machine`.<br>
What this means is that we are able to create multiple instances of `Ollama Client` as long as we have enough VRAM / RAM by exposing multiple ports on our `Host Machine` to connect to an isolated `Ollama Client`. Refer back to `Run Inferer` on how both the `Docker Container` and the `Inferer` should be linked to support multiple clients.
- `--name ollama_1` - Creates a Docker Container of name `ollama_1`
- `ollama/ollama` - Pulls the [Official Docker Image for Ollama](https://hub.docker.com/r/ollama/ollama)
- For more details and flags, refer to - [Official Docker Create Documentation](https://docs.docker.com/reference/cli/docker/container/create/)

### Starting the Docker Container

```bash
# After creating the Docker Container in the previous step
docker start ollama_1
```

### Pull a LLM Model from Ollama

```bash
# After Docker Container has been started in the previous step
docker exec ollama_1 ollama pull <model_name>
# Verify success - Should list all Ollama Models pulled across the 'ollama' volume
docker exec ollama_1 ollama list
```

### Terminating the Docker Container

```bash
# Stop the Docker Container when done
docker stop ollama_1
```

### Advanced Ollama Model Configurations

In order to configure a LLM Model, we have to first set up a `Modelfile`.

Preview an existing Model's Modelfile inside a Docker Container
```bash
# Access the volume as per normal, assuming that you used 'ollama pull <model_name>'
docker exec ollama_1 ollama show --modelfile <model_name>
```

```bash
# The Modelfile should look something similar to this
FROM /root/.ollama/models/custom-gguf/Mistral-Nemo-Instruct-2407.Q8_0.gguf
TEMPLATE """{{- range $i, $_ := .Messages }}
{{- if eq .Role "user" }}
{{- if and $.Tools (le (len (slice $.Messages $i)) 2) }}[AVAILABLE_TOOLS] {{ $.Tools }}[/AVAILABLE_TOOLS]
{{- end }}[INST] {{ if and $.System (eq (len (slice $.Messages $i)) 1) }}{{ $.System }}

{{ end }}{{ .Content }}[/INST]
{{- else if eq .Role "assistant" }}
{{- if .Content }} {{ .Content }}{{ if not (eq (len (slice $.Messages $i)) 1) }}</s>{{ end }}
{{- else if .ToolCalls }}[TOOL_CALLS] [
{{- range .ToolCalls }}{"name": "{{ .Function.Name }}", "arguments": {{ .Function.Arguments }}}
{{- end }}]</s>
{{- end }}
{{- else if eq .Role "tool" }}[TOOL_RESULTS] {"content": {{ .Content }}} [/TOOL_RESULTS]
{{- end }}
{{- end }}"""
PARAMETER stop [INST]
PARAMETER stop [/INST]
...
```

Simply copy the relevant sections of the Modelfile from the existing Model and create your own Modelfile using `vim`, or any other text editor. Subsequently, add any relevant `PARAMETERS` or `configurations` which you may need.

[Official Documentation](https://github.com/ollama/ollama/blob/main/docs/modelfile.md#template)

```bash
# Add any additional PARAMETERS that you may need in order to customize the Model
# E.g. num_ctx / temperature / top_k / top_p / min_p
PARAMETER num_ctx 8192
PARAMETER temperature 0

SYSTEM You are a Senior Software Engineer that will provide technical support to his Engineers through code examples and theoretical knowledge. If you are unsure of how to provide an answer, you will reply with 'I am sorry but I am unsure of the topic' instead of fabricating an answer.
...
```

In order to generate the actual `Model` from a `GGUF` file, 

```bash
# Creates a Model that can be referenced by Ollama
docker exec ollama_1 ollama create <your-desired-model-name> -f ./<your-modelfile-path>
# docker exec -it ollama_1 ollama create Mistral-Nemo-Instruct-2047-Q6-Ctx8192 -f ./root/.ollama/models/custom-gguf/Modelfile
```

Complete documentation for customizing [Ollama Modelfiles](https://github.com/ollama/ollama/blob/main/docs/modelfile.md#template)