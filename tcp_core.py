import re
import os
import sys
import json
import pickle
import socket
import asyncio
from ollama import AsyncClient
from sentence_transformers import SentenceTransformer, util
from utility import KnnSearch, SemanticSearch, FindValidFilesInDirectory

# Need for first run
# import nltk


# Tcp Server to receive and pass out the llm response
class TcpServer:
    def __init__(self, host, port, ollama_port):
        """
        Initialize the TcpServer instance.

        Args:
            host (str): The host IP address.
            port (int): The port number.
            ollama_port (str): The port number for the ollama client.
        """
        self.host = host
        self.port = port
        self.server = None
        self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        self.embeddings = None
        self.substring = "<doudousmacksoba>"
        self.async_client = AsyncClient("localhost:" + ollama_port)

        print("Host: ", host, "  |  Port: ", port)

    async def await_user_prompt(self, reader):
        """
        Wait for user input from the frontend.

        Args:
            reader (asyncio.StreamReader): The reader object.

        Returns:
            str: The user's prompt.
        """
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
        """
        Compress the entire conversation + latest user prompt into a single prompt.

        Args:
            msg (str): The user's prompt.

        Returns:
            str: The compressed prompt.
        """
        msg = json.loads(msg)
        _question_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question and do not add any additional text.
        Chat History:
        {}
        Follow Up Question:
        {}
        Standalone Question:"""

        chat_history = "\n".join(["{}: {}".format(m["role"], m["content"]) for m in msg[:-1]])
        follow_up_question = "{}: {}".format(msg[-1]["role"], msg[-1]["content"])

        response = await self.async_client.generate(model="llama3-temp0-Q6", prompt=_question_template.format(chat_history, follow_up_question), stream=False)
        parts = response['response'].split(":\n\n")
        return parts[1] if len(parts) > 1 else response['response']

    async def requires_rag(self, msg):
        """
        Check if RAG is required based on the latest compressed prompt.

        Args:
            msg (str): The compressed prompt.

        Returns:
            bool: True if RAG is required, False otherwise.
        """
        prompt_index = msg.rfind('"content":')
        # shift past content:
        prompt_index += 10

        _system_prompt = """Provide ONLY a single word response of Yes or No.
        Respond with Yes ONLY if you are able to provide an accurate and full answer and you are ABSOLUTELY CERTAIN.
        Respond with No otherwise."""

        response = await self.async_client.generate(model="llama3-temp0-Q6", prompt=msg[prompt_index:len(msg)-2], system=_system_prompt, stream=False)
        print("LLM is capable of providing an accurate response: ", response['response'])
        return True if response['response'] == "No" else False
    
    def display_ollama_performance(self, part):
        """
        Display the performance of the ollama client.

        Args:
            part (dict): The response from the ollama client.
        """
        token_count = part['eval_count']
        time_ns = part['eval_duration']
        print("Tokens per second: ", int(token_count) / (float(time_ns) / (1 * 10**9)))
        print("Token Count: ", int(token_count))
        print("Time (s): ", int(time_ns) / (1 * 10**9))

    async def handle_client(self, reader, writer):
        """
        Handle a client connection.

        Args:
            reader (asyncio.StreamReader): The reader object.
            writer (asyncio.StreamWriter): The writer object.
        """
        print("Received a message!")
        
        # Wait for user input from frontend
        msg = await self.await_user_prompt(reader)

        # Compress the entire conversation + latest user prompt into a single prompt
        compressed_prompt = await self.compress_convo(msg)

        print("Compressed prompt: ", compressed_prompt, "\n")

        # Check if RAG is required based on the latest compressed prompt
        requires_rag = await self.requires_rag(compressed_prompt)
        if requires_rag:
            # Remove starting brace "{" and encode user prompt for knn
            #msg = msg[1:]
            prompt_index = msg.rfind('"content":')
            question_embedding = self.model.encode([msg[prompt_index:len(msg)-2]])
            #best_matches = KnnSearch(question_embedding, all_embeddings, k=5) # Play with this value, higher seems to result in inaccurate results
            best_matches = SemanticSearch(question_embedding, self.embeddings, 5)

            # Construct the embedding text
            sourcetext = ""
            for i, (source_text, source_location) in enumerate(best_matches, start=1):
                #sourcetext += f"{i}. Index: {index}, Source Text: {source_text}, Source Location: {source_location}"
                sourcetext += f"{i}. Source Text: {source_text}, Source Location: {source_location}"
            
            # # Add on the system prompt to the front of the msg
            # sys_prompt_str = '[{"role": "system", "content": "Only use the provided context to answer the question. Context: ' + sourcetext + '"},'
            # msg = sys_prompt_str + msg
            # print("Concatenated message before json:\n", msg)

        data = json.loads(msg)
        if requires_rag:
            print("Received Message: ", msg)
            data[-1]['content'] = f'Using this data: {sourcetext}. Respond to this prompt: {compressed_prompt}. The response should also contain the Source Location where the response was obtained from.'
        print("Concatenated message after updating:\n", data)

        # Disabling Nagle's algorithm for the socket
        writer.get_extra_info('socket').setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        # Stream the LLM response
        async for part in await self.async_client.chat(model='Mistral-Nemo-Instruct-2047-Q6-Ctx8192', messages=data, stream=True, keep_alive="15m"):
            temp = part['message']['content']
            writer.write(temp.encode('utf-8'))
            await writer.drain()

        # Display Token Speed
        self.display_ollama_performance(part)
        
        writer.close()
        await writer.wait_closed()

    async def run(self):
        """
        Starts running the inferer indefinitely
        """
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

    all_embeddings = []
    pkl_list = FindValidFilesInDirectory(os.path.dirname(__file__) + '\\embedding')
    for pkl_file in pkl_list:
        with open(pkl_file, 'rb') as f:
            all_embeddings.append(pickle.load(f))
            f.close()
    
    svr.embeddings = all_embeddings # new
    print("Server started")
    asyncio.run(svr.run())