import asyncio
import json
import os
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession
from mcp.client.sse import sse_client

from dotenv import load_dotenv
from openai import OpenAI  # 修改为 OpenAI SDK


load_dotenv()  # load environment variables from .env

class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        # 初始化 OpenAI 客户端
        self.client = OpenAI(
            api_key="sk-e47d26f84c3c41b3a49c45a7a09fac40",  # 建议改为从环境变量读取
            base_url="https://api.deepseek.com/v1"
        )

    async def connect_to_sse_server(self, server_url: str):
        """Connect to an MCP server running with SSE transport"""
        # Store the context managers so they stay alive
        self._streams_context = sse_client(url=server_url)
        streams = await self._streams_context.__aenter__()

        self._session_context = ClientSession(*streams)
        self.session: ClientSession = await self._session_context.__aenter__()

        # Initialize
        await self.session.initialize()

        # List available tools to verify connection
        print("Initialized SSE client...")
        print("Listing tools...")
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    async def cleanup(self):
        """Properly clean up the session and streams"""
        if self._session_context:
            await self._session_context.__aexit__(None, None, None)
        if self._streams_context:
            await self._streams_context.__aexit__(None, None, None)

    async def call_deepseek_api(self, messages, tools=None):
        print(f"【mcp_message】: {messages}\n")
        print(f"【tools】: {tools}\n")
        """使用 OpenAI SDK 调用 DeepSeek"""
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                tools=tools or None,
                temperature=0.3,
                stream=False
            )
            return response
        except Exception as e:
            print(f"DeepSeek API Error: {str(e)}")
            return None
 
    async def process_query(self, query: str) -> str:
        """处理查询逻辑（适配 OpenAI SDK）"""
        messages = [{"role": "user", "content": query}]
        
        # 获取工具列表（格式需要适配）
        tool_response = await self.session.list_tools()
        available_tools = [{
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema
            }
        } for tool in tool_response.tools]
        #print(f"[可用工具 available_tools:{available_tools} ") 
        # 首次调用
        response = await self.call_deepseek_api(
            messages=messages,
            tools=available_tools if available_tools else None
        )
        print(f"[首次调用 response:{response} \n") 
        if not response:
            return "Failed to get response from DeepSeek"

        final_text = []
        message = response.choices[0].message

        # 处理文本响应
        if message.content:
            final_text.append(message.content)


        # 处理工具调用
        if message.tool_calls:
            for index, tool_call in enumerate(message.tool_calls):
                print(f"{tool_call}\n")
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                # 执行工具调用
                result = await self.session.call_tool(tool_name, tool_args)
                final_text.append(f"[调用工具 {tool_name} 参数：{tool_args}]\n")
    

                # 确保所有内容都是字符串类型
                tool_content = str(result.content) if result.content else ""

                # 先收集所有工具调用
                all_tool_calls = []
                all_tool_calls.append({
                    "id": tool_call.id,
                    "type": tool_call.type,
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    },
                    "index": index  # 获取索引，如果不存在则默认为0
                })                
                # 更新对话历史
                messages.append({
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": all_tool_calls
                })
                messages.append({
                    "role": "tool",
                    "content": tool_content,
                    "tool_call_id": tool_call.id
                })
                
            # 获取后续响应
            follow_up = await self.call_deepseek_api(messages)
            if follow_up:
                final_text.append(follow_up.choices[0].message.content)

        return "\n".join(final_text)
    

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")
        
        while True:
            try:
                query = input("\nQuery: ").strip()
                
                if query.lower() == 'quit':
                    break
                    
                response = await self.process_query(query)
                print("\n" + response)
                    
            except Exception as e:
                print(f"\nError: {str(e)}")


async def main():
    '''
    if len(sys.argv) < 2:
        print("Usage: uv run client.py <URL of SSE MCP server (i.e. http://localhost:8080/sse)>")
        sys.exit(1)
    '''
    client = MCPClient()
    try:
        await client.connect_to_sse_server(server_url="http://127.0.0.1:8080/sse")
        await client.chat_loop()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    import sys
    asyncio.run(main())
    
    