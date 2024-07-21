import json
import asyncio
import socket
import sys
import re
import os
from ollama import AsyncClient
from sentence_transformers import SentenceTransformer
from custom_embedder import GenerateAllEmbeddings, KnnSearch

# Tcp Server to receive and pass out the llm response
class TcpServer:
    def __init__(self, host, port, ollama_port):
        self.host = host
        self.port = port
        self.ollama_port = ollama_port
        self.server = None
        self.model = None
        self.embeddings = None
        print("Host: ", host, "  |  Port: ", port)

    async def handle_client(self, reader, writer):
        print("msg received")

        buffer = bytearray()
        substring = "<doudousmacksoba>"

        # Get the complete message first to pass to llm
        while True:
            data = await reader.read(1024)
            string_value = data.decode('utf-8')
            buffer.extend(data)
            if substring in string_value:
                break
        
        # Interpret the User's Prompt and remove the unique terminator
        msg = buffer.decode()
        msg = msg[:-len(substring)]
        msg = msg[1:]
        # need to find a way to extract the latest question
        question_embedding = model.encode(["who is king ryan of ryry land?"])   
        best_matches = KnnSearch(question_embedding, all_embeddings, k=5) # Play with this value, higher seems to result in inaccurate results

        sourcetext = ""
        for i, (index, source_text) in enumerate(best_matches, start=1):
            sourcetext += f"{i}. Index: {index}, Source Text: {source_text}"
        
        sys_prompt_str = '[{"role": "system", "content": "Use this information if you are unable to provide an answer:' + sourcetext + '"},'
        msg = sys_prompt_str + msg
        data = json.loads(msg)

        # Disabling Nagle's algorithm for the socket
        writer.get_extra_info('socket').setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        host_string = "localhost:" + self.ollama_port
        async_client = AsyncClient(host=host_string)

        # tinyllama:1.1b-chat-v0.6-q6_K
        # llama3:8b-instruct-q6_K
        async for part in await async_client.chat(model='llama3:8b-instruct-q6_K', messages=data, stream=True):
            writer.write(part['message']['content'].encode('utf-8'))
            await writer.drain()
            #print(part['message']['content'], end='', flush=True)
        
        token_count = part['eval_count']
        time_ns = part['eval_duration']
        print("Tokens per second: ", int(token_count) / (float(time_ns) / (1 * 10**9)))
        print("Token Count: ", int(token_count))
        print("Time (s): ", int(time_ns) / (1 * 10**9))

        writer.close()
        await writer.wait_closed()

    async def run(self):
        self.server = await asyncio.start_server(self.handle_client, self.host, self.port)

        async with self.server:
            await self.server.serve_forever()

if __name__== "__main__":
    print("Server started")
    model = SentenceTransformer('all-MiniLM-L6-v2')
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
    svr.model = model
    all_embeddings = GenerateAllEmbeddings(os.path.dirname(__file__) + '\\compressed\\', model)
    svr.embeddings = all_embeddings
    asyncio.run(svr.run())