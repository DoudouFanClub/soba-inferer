import asyncio
import ollama

# Tcp Server to receive and pass out the llm response
class TcpServer:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.server = None

    async def handle_client(self, reader, writer):
        addr = writer.get_extra_info('peername')

        buffer = bytearray()

        # Get the complete message first to pass to llm
        while True:
            data = await reader.read(1024)
            if not data:
                break
            buffer.extend(data)
        
        msg = buffer.decode()

        stream = ollama.chat(
            model = msg.model,
            messages = msg.messages,
            stream = True
        )

        # Write out the stream back to the client
        for chunk in stream:
            writer.write(chunk['message']['content'])

        writer.close()

    async def run(self):
        self.server = await asyncio.start_server(self.handle_client, self.host, self.port)

        async with self.server:
            await self.server.serve_forever()

if __name__== "__main__":
    svr = TcpServer("localhost", 7060)
    asyncio.run(svr.run())