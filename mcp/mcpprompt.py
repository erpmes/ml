#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 21 10:12:50 2025

@author: gaffliu
"""

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
        #print(f"【tools】: {tools}\n")
        """使用 OpenAI SDK 调用 DeepSeek"""
        try:
            response = self.client.chat.completions.create(
                model="deepseek-reasoner",
                messages=messages,
                temperature=0.3,
                stream=False
            )
            return response
        except Exception as e:
            print(f"DeepSeek API Error: {str(e)}")
            return None
 
    async def process_query(self, query: str) -> str:
        
        # 获取工具列表（格式需要适配）
        tool_response = await self.session.list_tools()
        
        available_tools = ""

        for tool in tool_response.tools:
            available_tools = available_tools + f"function_name: {tool.name}, function_description: {tool.description} ,function_parameters: {tool.inputSchema} ;"
        #print(f"[可用工具 available_tools:{available_tools} ") 
        
        system_prompt = (
        "您是一位得力助手。"
        "您拥有使用工具的功能。"
        "请根据【用户问题】，来判断是否使用【工具列表】中的工具。"
        "使用工具时请勿丢失用户的问题信息，"
        "并尽量保持问题内容的完整性。"
        "如果需要使用工具请在think阶段明确调用方式"
        "调用工具情使用格式化输出，输出格式如下：'tool_calls': [ChatCompletionMessageToolCall(id='call_0_a4cb9205-9ffb-4933-baa0-5f1dc0fe628e', function=Function(arguments='{\"content\": \"【用户问题】\"}', name='search_seasons_knowledge'), type='function', index=0), ChatCompletionMessageToolCall(id='call_1_403a4b18-e108-48af-9657-a5dfbce83a51', function=Function(arguments='{\"end_time\": \"2023/12/31\"}', name='activity_get_by_time'), type='function', index=1)]}" 
        "如果不需要使用工具请直接回答用户问题"
        
        )
        
        
        """处理查询逻辑（适配 OpenAI SDK）"""
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": "【用户问题】："+query+"，【工具列表】："+available_tools}]
        
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
            final_text.append("【think:】\n")
            final_text.append(message.reasoning_content)
            final_text.append("【anwser:】\n")
            final_text.append(message.content)
        '''
        # 处理工具调用
        if message.tool_calls:
            for tool_call in message.tool_calls:
                print(f"{tool_call}\n")
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                # 执行工具调用
                result = await self.session.call_tool(tool_name, tool_args)
                final_text.append(f"[调用工具 {tool_name} 参数：{tool_args}]\n")
    

                # 确保所有内容都是字符串类型
                tool_content = str(result.content) if result.content else ""

                # 更新对话历史
                messages.append({
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": message.tool_calls
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
        '''
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
        await client.connect_to_sse_server(server_url="http://deepsearcher-mcp.gzp-in.woa.com/sse")
        await client.chat_loop()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    import sys
    asyncio.run(main())
    
    