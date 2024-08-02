import json
import asyncio
import socket
import sys
import re
import os
# need for first runn
# import nltk 
from ollama import AsyncClient
from sentence_transformers import SentenceTransformer
from custom_embedder import GenerateAllEmbeddings, KnnSearch
import pickle

# Tcp Server to receive and pass out the llm response
class TcpServer:
    def __init__(self, host, port, ollama_port):
        self.host = host
        self.port = port
        self.server = None
        self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        self.embeddings = None
        self.substring = "<doudousmacksoba>"
        self.async_client = AsyncClient("localhost:" + ollama_port)

        print("Host: ", host, "  |  Port: ", port)

    async def await_user_prompt(self, reader):
        buffer = bytearray()
        while True:
            data = await reader.read(1024)
            string_value = data.decode('utf-8')
            buffer.extend(data)
            if self.substring in string_value:
                # Interpret the User's Prompt and remove the unique terminator
                msg = buffer.decode()
                msg = msg[:-len(self.substring)]
                return msg
    
    async def compress_convo(self, msg):
        msg = json.loads(msg)
        _question_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
        Chat History:
        {}
        Follow Up Question:
        {}
        Standalone Question:"""

        chat_history = "\n".join(["{}: {}".format(m["role"], m["content"]) for m in msg[:-1]])
        follow_up_question = "{}: {}".format(msg[-1]["role"], msg[-1]["content"])

        response = await self.async_client.generate(model="llama3-temp0-Q6", prompt=_question_template.format(chat_history, follow_up_question), stream=False)
        # print("Question template: ", _question_template.format(chat_history, follow_up_question), "\n")
        return response['response']

    async def requires_rag(self, msg):
        prompt_index = msg.rfind('"content":')
        # shift past content:
        prompt_index += 10

        _system_prompt = """Provide ONLY a single word response of Yes or No.
        Respond with Yes ONLY if you are able to provide an accurate and full answer and you are ABSOLUTELY CERTAIN.
        Respond with No otherwise."""

        # remove the special characters from behind
        response = await self.async_client.generate(model="llama3-temp0-Q6", prompt=msg[prompt_index:len(msg)-2], system=_system_prompt, stream=False)

        print("Requires rag response: ", response['response'])

        return True if response['response'] == "No" else False
    
    def display_ollama_performance(self, part):
        token_count = part['eval_count']
        time_ns = part['eval_duration']
        print("Tokens per second: ", int(token_count) / (float(time_ns) / (1 * 10**9)))
        print("Token Count: ", int(token_count))
        print("Time (s): ", int(time_ns) / (1 * 10**9))

    async def handle_client(self, reader, writer):
        print("msg received")
        
        # Wait for user input from frontend
        msg = await self.await_user_prompt(reader)

        # Compress the entire conversation + latest user prompt into a single prompt
        compressed_prompt = await self.compress_convo(msg)

        print("Compressed prompt: ", compressed_prompt, "\n")

        # Check if RAG is required based on the latest compressed prompt
        if await self.requires_rag(compressed_prompt):
            print("User's prompt requires RAG")

            # Remove starting brace "{" and encode user prompt for knn
            msg = msg[1:]
            prompt_index = msg.rfind('"content":')
            question_embedding = self.model.encode([msg[prompt_index:len(msg)-2]])
            best_matches = KnnSearch(question_embedding, all_embeddings, k=5) # Play with this value, higher seems to result in inaccurate results

            # Construct the embedding text
            sourcetext = ""
            for i, (index, source_text) in enumerate(best_matches, start=1):
                sourcetext += f"{i}. Index: {index}, Source Text: {source_text}"
            
            # Add on the system prompt to the front of the msg
            _system_prompt = """Try your best to answer the question with your own knowledge and the context of the conversation before referencing the System Prompt section.
            Follow the rules listed below when providing a response.

            Rules:
            1) If the question includes a request for code, provide a code block directly from the System Prompt section.
            2) If you do not have the information to a question, just say this - Hmm, I'm not too sure. Do not make up an answer.
            3) If the question includes a request for text information, attempt to answer the question with the System Prompt.
            4) If you have information on where the code example or text information was obtained from, provide it at the end of the response.

            System Prompt: """
            sys_prompt_str = '[{"role": "system", "content": "Try your best to answer the questions with your own knowledge and the context of the conversation, only if that is not possible then use the following information:' + sourcetext + '"},'
            msg = sys_prompt_str + msg
            print("Concatenated message before json:\n", msg)

        data = json.loads(msg)

        # Disabling Nagle's algorithm for the socket
        writer.get_extra_info('socket').setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        # Stream the LLM response
        async for part in await self.async_client.chat(model='mistral-nemo:12b-instruct-2407-q6_K', messages=data, stream=True, keep_alive="15m"): # llama3:8b-instruct-q6_K    |   mistral-nemo:12b-instruct-2407-q6_K    |    llama3.1:8b-instruct-q8_0
            temp = part['message']['content']
            writer.write(temp.encode('utf-8'))
            await writer.drain()

        # Display Token Speed
        self.display_ollama_performance(part)
        
        writer.close()
        await writer.wait_closed()

    async def run(self):
        self.server = await asyncio.start_server(self.handle_client, self.host, self.port)

        async with self.server:
            await self.server.serve_forever()

if __name__== "__main__":
    # needed for first run
    #nltk.download('punkt')
    def validate_ip(ip):
        ip_pattern = r"^((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$"
        if re.match(ip_pattern, ip):
            return True
        return False
    if validate_ip(sys.argv[1]):
        print("Valid IP address")
    else:
        print("Invalid IP address", sys.argv[1])
        exit()
    svr = TcpServer(sys.argv[1], sys.argv[2], sys.argv[3])

    all_embeddings = ""
    with open(os.path.dirname(__file__) + '\\embeddings.pkl', 'rb') as f:
        all_embeddings = pickle.load(f)
    
    svr.embeddings = all_embeddings
    print("Server started")
    asyncio.run(svr.run())