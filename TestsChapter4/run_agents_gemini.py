"""
Designing Multi-Agent Systems — Ch. 4 · Code Along v1→v4
Adapted to use Google Gemini API instead of Anthropic.

Requirements:
    pip install google-genai

Usage (Windows):
    set GEMINI_API_KEY=AIzaSy...
    python run_agents_gemini.py

    python run_agents_gemini.py --v 2    # only v2 (tools)
    python run_agents_gemini.py --v 4    # only v4 (streaming)
"""

import asyncio
import inspect
import json
import sys
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Union

from google import genai
from google.genai import types

import os
API_KEY = os.environ.get("GEMINI_API_KEY", "")
if not API_KEY:
    print("ERROR: GEMINI_API_KEY not set. Run: set GEMINI_API_KEY=AIzaSy...")
    sys.exit(1)

client = genai.Client(api_key=API_KEY)
DEFAULT_MODEL = "gemini-2.0-flash"

# ─────────────────────────────────────────────
# Shared data structures (same as picoagents)
# ─────────────────────────────────────────────

@dataclass
class Message:
    content: str
    source: str = "assistant"

@dataclass
class ToolMessage(Message):
    source: str = "tool"
    tool_name: str = ""

@dataclass
class AgentResponse:
    messages: List[Message] = field(default_factory=list)
    source: str = ""

    @property
    def final_content(self) -> str:
        for msg in reversed(self.messages):
            if msg.source == "assistant" and msg.content:
                return msg.content
        return ""

@dataclass
class ToolCallEvent:
    tool_name: str
    parameters: Dict[str, Any]
    source: str = ""

@dataclass
class ToolResultEvent:
    tool_name: str
    result: str
    source: str = ""

# ─────────────────────────────────────────────
# Helper: Python function → Gemini FunctionDeclaration
# ─────────────────────────────────────────────

def _get_type_string(annotation) -> str:
    return {str: "STRING", int: "INTEGER", float: "NUMBER", bool: "BOOLEAN"}.get(annotation, "STRING")

def _function_to_gemini_tool(func: Callable) -> types.Tool:
    sig = inspect.signature(func)
    doc = inspect.getdoc(func) or ""
    properties = {}
    required = []
    for name, param in sig.parameters.items():
        t = "STRING"
        if param.annotation != inspect.Parameter.empty:
            t = _get_type_string(param.annotation)
        properties[name] = types.Schema(type=t, description=f"Parameter {name}")
        if param.default == inspect.Parameter.empty:
            required.append(name)
    declaration = types.FunctionDeclaration(
        name=func.__name__,
        description=doc,
        parameters=types.Schema(
            type="OBJECT",
            properties=properties,
            required=required
        )
    )
    return types.Tool(function_declarations=[declaration])

# ─────────────────────────────────────────────
# Memory (v3)
# ─────────────────────────────────────────────

@dataclass
class MemoryItem:
    content: str

class ListMemory:
    """Simple list memory — same interface as picoagents.memory.ListMemory."""

    def __init__(self, max_memories: int = 100):
        self.memories: List[MemoryItem] = []
        self.max_memories = max_memories

    async def add(self, content: str) -> None:
        self.memories.append(MemoryItem(content=content))
        if len(self.memories) > self.max_memories:
            self.memories = self.memories[-self.max_memories:]

    async def get_context(self, max_items: int = 10) -> List[str]:
        return [m.content for m in self.memories[-max_items:]]

# ─────────────────────────────────────────────
# V1 — Basic agent
# ─────────────────────────────────────────────

class AgentV1:
    """Ch 4.1 · Minimal loop: task → LLM → response."""

    def __init__(self, name: str,
                 instructions: str = "You are a helpful assistant.",
                 model: str = DEFAULT_MODEL):
        self.name = name
        self.instructions = instructions
        self.model = model

    async def run(self, task: str) -> AgentResponse:
        response = client.models.generate_content(
            model=self.model,
            config=types.GenerateContentConfig(system_instruction=self.instructions),
            contents=task
        )
        content = response.text or ""
        return AgentResponse(
            messages=[
                Message(content=task, source="user"),
                Message(content=content, source="assistant")
            ],
            source=self.name
        )

# ─────────────────────────────────────────────
# V2 — Agent with tools
# ─────────────────────────────────────────────

class AgentV2:
    """Ch 4.2 · Tool calling loop with Python functions."""

    def __init__(self, name: str,
                 instructions: str = "You are a helpful assistant.",
                 model: str = DEFAULT_MODEL,
                 tools: Optional[List[Callable]] = None,
                 max_iterations: int = 10):
        self.name = name
        self.instructions = instructions
        self.model = model
        self.max_iterations = max_iterations
        self._tools: Dict[str, Callable] = {}
        self._gemini_tools: List[types.Tool] = []
        if tools:
            for t in tools:
                self._tools[t.__name__] = t
                self._gemini_tools.append(_function_to_gemini_tool(t))

    def _execute_tool(self, name: str, args: Dict[str, Any]) -> str:
        if name in self._tools:
            try:
                return str(self._tools[name](**args))
            except Exception as e:
                return f"Error executing {name}: {e}"
        return f"Tool '{name}' not found."

    async def run(self, task: str) -> AgentResponse:
        all_messages: List[Message] = [Message(content=task, source="user")]
        contents = [types.Content(role="user", parts=[types.Part(text=task)])]

        for _ in range(self.max_iterations):
            response = client.models.generate_content(
                model=self.model,
                config=types.GenerateContentConfig(
                    system_instruction=self.instructions,
                    tools=self._gemini_tools if self._gemini_tools else None
                ),
                contents=contents
            )

            # Check for function calls
            fn_calls = []
            for part in response.candidates[0].content.parts:
                if part.function_call:
                    fn_calls.append(part.function_call)

            if not fn_calls:
                content = response.text or ""
                all_messages.append(Message(content=content, source="assistant"))
                return AgentResponse(messages=all_messages, source=self.name)

            # Add model response to history
            contents.append(response.candidates[0].content)

            # Execute tools and build function response parts
            fn_response_parts = []
            for fn_call in fn_calls:
                name = fn_call.name
                args = dict(fn_call.args)
                print(f"  [tool] {name}({args})")
                result = self._execute_tool(name, args)
                print(f"  [result] {result}")
                all_messages.append(ToolMessage(content=result, tool_name=name))
                fn_response_parts.append(
                    types.Part(
                        function_response=types.FunctionResponse(
                            name=name,
                            response={"result": result}
                        )
                    )
                )

            contents.append(types.Content(role="user", parts=fn_response_parts))

        all_messages.append(Message(content="Max iterations reached.", source="assistant"))
        return AgentResponse(messages=all_messages, source=self.name)

# ─────────────────────────────────────────────
# V3 — Agent with memory
# ─────────────────────────────────────────────

class AgentV3(AgentV2):
    """Ch 4.3 · Memory persists across run() calls."""

    def __init__(self, *args, memory: Optional[ListMemory] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._memory = memory or ListMemory()

    async def run(self, task: str) -> AgentResponse:
        ctx = await self._memory.get_context()
        memory_section = ""
        if ctx:
            memory_section = "\n\nMemory context (previous interactions):\n" + "\n".join(f"- {c}" for c in ctx)

        original = self.instructions
        self.instructions = self.instructions + memory_section
        response = await super().run(task)
        self.instructions = original

        await self._memory.add(f"User: {task}")
        await self._memory.add(f"Assistant: {response.final_content}")
        return response

# ─────────────────────────────────────────────
# V4 — Agent with streaming
# ─────────────────────────────────────────────

class AgentV4(AgentV3):
    """Ch 4.4 · Streaming: emits events in real time."""

    async def run_stream(self, task: str) -> AsyncGenerator[Union[ToolCallEvent, ToolResultEvent, AgentResponse], None]:
        ctx = await self._memory.get_context()
        memory_section = ""
        if ctx:
            memory_section = "\n\nMemory context:\n" + "\n".join(f"- {c}" for c in ctx)
        system = self.instructions + memory_section

        all_messages: List[Message] = [Message(content=task, source="user")]
        contents = [types.Content(role="user", parts=[types.Part(text=task)])]

        for _ in range(self.max_iterations):
            full_text = ""
            fn_calls = []

            # Streaming call
            with client.models.generate_content_stream(
                model=self.model,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    tools=self._gemini_tools if self._gemini_tools else None
                ),
                contents=contents
            ) as stream:
                for chunk in stream:
                    for part in chunk.candidates[0].content.parts:
                        if part.text:
                            full_text += part.text
                            print(part.text, end="", flush=True)
                        if part.function_call:
                            fn_calls.append(part.function_call)

                final = stream.get_final_response()

            # Re-check function calls from final response
            if not fn_calls:
                for part in final.candidates[0].content.parts:
                    if part.function_call:
                        fn_calls.append(part.function_call)

            if not fn_calls:
                all_messages.append(Message(content=full_text, source="assistant"))
                await self._memory.add(f"User: {task}")
                await self._memory.add(f"Assistant: {full_text}")
                yield AgentResponse(messages=all_messages, source=self.name)
                return

            contents.append(final.candidates[0].content)
            fn_response_parts = []

            for fn_call in fn_calls:
                name = fn_call.name
                args = dict(fn_call.args)
                yield ToolCallEvent(tool_name=name, parameters=args, source=self.name)
                result = self._execute_tool(name, args)
                yield ToolResultEvent(tool_name=name, result=result, source=self.name)
                all_messages.append(ToolMessage(content=result, tool_name=name))
                fn_response_parts.append(
                    types.Part(
                        function_response=types.FunctionResponse(
                            name=name,
                            response={"result": result}
                        )
                    )
                )

            contents.append(types.Content(role="user", parts=fn_response_parts))

# ─────────────────────────────────────────────
# Demos
# ─────────────────────────────────────────────

SEP = "\n" + "─" * 55 + "\n"

async def demo_v1():
    print(SEP + "V1 · Basic agent" + SEP)
    agent = AgentV1(name="assistant",
                    instructions="You are a concise and helpful assistant.")
    response = await agent.run("What is the capital of France and why is it important?")
    print(f"Agent: {response.final_content}")

async def demo_v2():
    print(SEP + "V2 · Agent with tools" + SEP)

    def get_weather(location: str) -> str:
        """Get current weather for a location."""
        data = {"Bilbao": "16°C, cloudy", "Madrid": "24°C, sunny", "Barcelona": "21°C, partly cloudy"}
        return data.get(location, f"22°C, clear in {location}")

    def calculate(expression: str) -> str:
        """Evaluate a safe math expression."""
        try:
            allowed = set("0123456789+-*/()., ")
            if not all(c in allowed for c in expression):
                return "Expression not allowed"
            return f"{expression} = {eval(expression)}"
        except Exception as e:
            return f"Error: {e}"

    agent = AgentV2(
        name="assistant",
        instructions="You are a helpful assistant. Use tools when needed.",
        tools=[get_weather, calculate]
    )
    print("Question: What's the weather in Bilbao and what is 15 * 24?\n")
    response = await agent.run("What's the weather in Bilbao and what is 15 * 24?")
    print(f"\nAgent: {response.final_content}")

async def demo_v3():
    print(SEP + "V3 · Agent with memory" + SEP)
    memory = ListMemory()
    agent = AgentV3(
        name="assistant",
        instructions="You are a helpful assistant with memory.",
        memory=memory
    )
    print("Turn 1:")
    r1 = await agent.run("My name is Silvia and I am an AI expert at Iberdrola, in Bilbao.")
    print(f"Agent: {r1.final_content}\n")

    print("Turn 2 (agent should remember the context):")
    r2 = await agent.run("Do you remember who I am and where I work?")
    print(f"Agent: {r2.final_content}")

async def demo_v4():
    print(SEP + "V4 · Real-time streaming" + SEP)

    def get_time(city: str) -> str:
        """Get current local time for a city."""
        return f"14:32 CEST in {city}"

    agent = AgentV4(
        name="assistant",
        instructions="You are an expert in multi-agent systems.",
        tools=[get_time]
    )

    print("Question: What are multi-agent systems?\n")
    print("Streaming → ", end="")

    async for event in agent.run_stream("Explain in 3 key points what multi-agent systems are and why they are useful."):
        if isinstance(event, ToolCallEvent):
            print(f"\n  [tool_call] {event.tool_name}({event.parameters})")
        elif isinstance(event, ToolResultEvent):
            print(f"  [tool_result] {event.result}\n  → ", end="")
        elif isinstance(event, AgentResponse):
            print(f"\n\nFull response ({len(event.final_content)} chars)")

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

async def main():
    arg = sys.argv[2] if len(sys.argv) > 2 and sys.argv[1] == "--v" else "all"
    if arg == "1" or arg == "all": await demo_v1()
    if arg == "2" or arg == "all": await demo_v2()
    if arg == "3" or arg == "all": await demo_v3()
    if arg == "4" or arg == "all": await demo_v4()

if __name__ == "__main__":
    asyncio.run(main())
