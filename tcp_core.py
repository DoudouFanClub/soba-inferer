import asyncio
import socket
import sys
import time
import ollama
from ollama import AsyncClient

# Tcp Server to receive and pass out the llm response
class TcpServer:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.server = None

    async def handle_client(self, reader, writer):
        addr = writer.get_extra_info('peername')
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
        print("received this: ", msg)
        msg = msg[:-len(substring)]

        # Disabling Nagle's algorithm for the socket
        writer.get_extra_info('socket').setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)


        async for part in await AsyncClient().chat(model='phi3:mini', messages=[
                {
                    "role": "user",
                    "content": msg
                }
            ], stream=True):
            writer.write(part['message']['content'].encode('utf-8'))
            await writer.drain()
            print(part['message']['content'], end='', flush=True)

        # # Send the Query to the phi3:mini LLM Model and retrieve
        # # a dynamically updated stream
        # stream = ollama.chat(
        #     model="phi3:mini",
        #     messages=[
        #         {
        #             "role": "user",
        #             "content": msg
        #         }
        #     ],
        #     stream=True
        # )

        # # Write out the stream back to the client
        # for chunk in stream:
        #     writer.write(chunk['message']['content'].encode('utf-8'))
        #     await writer.drain()
        #     print(chunk['message']['content'], flush=True)

        writer.close()
        await writer.wait_closed()

    async def run(self):
        self.server = await asyncio.start_server(self.handle_client, self.host, self.port)

        async with self.server:
            await self.server.serve_forever()

if __name__== "__main__":
    svr = TcpServer("localhost", 7060)
    asyncio.run(svr.run())