# LLM Inferer
This will serve as the LLM Inferer that will serve as the endpoint for the LLM Server.

## Dependencies
Currently the inferer requires `ollama` in order to function, this should be replaced in the future either using `llama_cpp`, `pytorch` or `transformers` by Hugging Face.

## How to use
Currently just using the following commands should allow for testing of the LLM Server
```python
pip install ollama
python tcp_core.py
```
