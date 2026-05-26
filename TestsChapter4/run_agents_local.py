"""
Designing Multi-Agent Systems — Cap. 4 · Code Along v1→v4
Adaptado para usar Anthropic (Claude) en lugar de Azure OpenAI.

Requisitos:
    pip install anthropic

Uso:
    export ANTHROPIC_API_KEY="sk-ant-..."
    python run_agents.py          # ejecuta los 4 ejemplos
    python run_agents.py --v 2    # solo v2 (tools)
"""

import asyncio
import inspect
import json
import sys
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Union

import anthropic

# ─────────────────────────────────────────────
# Estructuras de datos compartidas
# ─────────────────────────────────────────────

@dataclass
class Message:
    content: str
    source: str = "assistant"

@dataclass
class ToolMessage(Message):
    source: str = "tool"
    tool_call_id: str = ""
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
# Helper: función Python → tool schema Anthropic
# ─────────────────────────────────────────────

def _get_type_string(annotation) -> str:
    return {str: "string", int: "integer", float: "number", bool: "boolean"}.get(annotation, "string")

def _function_to_schema(func: Callable) -> Dict[str, Any]:
    sig = inspect.signature(func)
    doc = inspect.getdoc(func) or ""
    properties, required = {}, []
    for name, param in sig.parameters.items():
        t = "string"
        if param.annotation != inspect.Parameter.empty:
            t = _get_type_string(param.annotation)
        properties[name] = {"type": t, "description": f"Parameter {name}"}
        if param.default == inspect.Parameter.empty:
            required.append(name)
    return {
        "name": func.__name__,
        "description": doc,
        "input_schema": {"type": "object", "properties": properties, "required": required}
    }

# ─────────────────────────────────────────────
# Memory (para v3)
# ─────────────────────────────────────────────

@dataclass
class MemoryItem:
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

class ListMemory:
    """Memoria simple en lista — mismo interface que picoagents.memory.ListMemory."""

    def __init__(self, max_memories: int = 100):
        self.memories: List[MemoryItem] = []
        self.max_memories = max_memories

    async def add(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        self.memories.append(MemoryItem(content=content, metadata=metadata or {}))
        if len(self.memories) > self.max_memories:
            self.memories = self.memories[-self.max_memories:]

    async def get_context(self, max_items: int = 10) -> List[str]:
        return [m.content for m in self.memories[-max_items:]]

# ─────────────────────────────────────────────
# V1 — Agente básico
# ─────────────────────────────────────────────

class AgentV1:
    """Cap 4.1 · Loop mínimo: task → LLM → respuesta."""

    def __init__(self, name: str, instructions: str = "You are a helpful assistant.",
                 model: str = "claude-sonnet-4-20250514"):
        self.name = name
        self.instructions = instructions
        self.model = model
        self._client = anthropic.AsyncAnthropic()

    async def run(self, task: str) -> AgentResponse:
        response = await self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=self.instructions,
            messages=[{"role": "user", "content": task}]
        )
        content = response.content[0].text
        return AgentResponse(
            messages=[Message(content=task, source="user"),
                      Message(content=content, source="assistant")],
            source=self.name
        )

# ─────────────────────────────────────────────
# V2 — Agente con tools
# ─────────────────────────────────────────────

class AgentV2:
    """Cap 4.2 · Tool calling loop con funciones Python."""

    def __init__(self, name: str, instructions: str = "You are a helpful assistant.",
                 model: str = "claude-sonnet-4-20250514",
                 tools: Optional[List[Callable]] = None, max_iterations: int = 10):
        self.name = name
        self.instructions = instructions
        self.model = model
        self.max_iterations = max_iterations
        self._client = anthropic.AsyncAnthropic()
        self._tools: Dict[str, Callable] = {}
        self._tool_schemas: List[Any] = []
        if tools:
            for t in tools:
                self._tools[t.__name__] = t
                self._tool_schemas.append(_function_to_schema(t))

    def _execute_tool(self, name: str, args: Dict[str, Any]) -> str:
        if name in self._tools:
            try:
                return str(self._tools[name](**args))
            except Exception as e:
                return f"Error executing {name}: {e}"
        return f"Tool '{name}' not found."

    async def run(self, task: str) -> AgentResponse:
        all_messages: List[Message] = [Message(content=task, source="user")]
        api_messages: List[Any] = [{"role": "user", "content": task}]

        for _ in range(self.max_iterations):
            kwargs = dict(model=self.model, max_tokens=1024,
                          system=self.instructions, messages=api_messages)
            if self._tool_schemas:
                kwargs["tools"] = self._tool_schemas

            response = await self._client.messages.create(**kwargs)

            # Sin tool calls → respuesta final
            if response.stop_reason != "tool_use":
                text_blocks = [b.text for b in response.content if hasattr(b, "text")]
                content = " ".join(text_blocks)
                all_messages.append(Message(content=content, source="assistant"))
                return AgentResponse(messages=all_messages, source=self.name)

            # Hay tool calls
            api_messages.append({"role": "assistant", "content": response.content})
            tool_results = []

            for block in response.content:
                if block.type == "tool_use":
                    name, args, tid = block.name, block.input, block.id
                    print(f"  [tool] {name}({args})")
                    result = self._execute_tool(name, args)
                    print(f"  [result] {result}")
                    all_messages.append(ToolMessage(content=result, tool_call_id=tid, tool_name=name))
                    tool_results.append({"type": "tool_result", "tool_use_id": tid, "content": result})

            api_messages.append({"role": "user", "content": tool_results})

        all_messages.append(Message(content="Max iterations reached.", source="assistant"))
        return AgentResponse(messages=all_messages, source=self.name)

# ─────────────────────────────────────────────
# V3 — Agente con memoria
# ─────────────────────────────────────────────

class AgentV3(AgentV2):
    """Cap 4.3 · Memoria entre llamadas a run()."""

    def __init__(self, *args, memory: Optional[ListMemory] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._memory = memory or ListMemory()

    async def run(self, task: str) -> AgentResponse:
        # Inyectar contexto de memoria en el system prompt
        ctx = await self._memory.get_context()
        memory_section = ""
        if ctx:
            memory_section = "\n\nMemory context (previous interactions):\n" + "\n".join(f"- {c}" for c in ctx)

        original_instructions = self.instructions
        self.instructions = self.instructions + memory_section

        response = await super().run(task)

        self.instructions = original_instructions

        # Guardar en memoria
        await self._memory.add(f"User: {task}")
        await self._memory.add(f"Assistant: {response.final_content}")

        return response

# ─────────────────────────────────────────────
# V4 — Agente con streaming
# ─────────────────────────────────────────────

class AgentV4(AgentV3):
    """Cap 4.4 · Streaming: emite eventos en tiempo real."""

    async def run_stream(self, task: str) -> AsyncGenerator[Union[ToolCallEvent, ToolResultEvent, AgentResponse], None]:
        ctx = await self._memory.get_context()
        memory_section = ""
        if ctx:
            memory_section = "\n\nMemory context:\n" + "\n".join(f"- {c}" for c in ctx)
        system = self.instructions + memory_section

        api_messages: List[Any] = [{"role": "user", "content": task}]
        all_messages: List[Message] = [Message(content=task, source="user")]

        for _ in range(self.max_iterations):
            kwargs = dict(model=self.model, max_tokens=1024,
                          system=system, messages=api_messages)
            if self._tool_schemas:
                kwargs["tools"] = self._tool_schemas

            # Streaming
            full_text = ""
            tool_uses = []

            async with self._client.messages.stream(**kwargs) as stream:
                async for event in stream:
                    # Texto en tiempo real
                    if hasattr(event, 'type'):
                        if event.type == 'content_block_delta':
                            if hasattr(event.delta, 'text'):
                                full_text += event.delta.text
                                print(event.delta.text, end="", flush=True)
                            elif hasattr(event.delta, 'partial_json'):
                                pass  # tool args streaming — ignoramos aquí

                final_msg = await stream.get_final_message()

            # Comprobar si hay tool_use en el mensaje final
            for block in final_msg.content:
                if block.type == "tool_use":
                    tool_uses.append(block)

            if not tool_uses:
                # Fin: no hay más tools
                all_messages.append(Message(content=full_text, source="assistant"))
                await self._memory.add(f"User: {task}")
                await self._memory.add(f"Assistant: {full_text}")
                yield AgentResponse(messages=all_messages, source=self.name)
                return

            # Hay tools — emitir eventos y ejecutar
            api_messages.append({"role": "assistant", "content": final_msg.content})
            tool_results = []

            for block in tool_uses:
                event_call = ToolCallEvent(tool_name=block.name, parameters=block.input, source=self.name)
                yield event_call

                result = self._execute_tool(block.name, block.input)
                event_result = ToolResultEvent(tool_name=block.name, result=result, source=self.name)
                yield event_result

                all_messages.append(ToolMessage(content=result, tool_call_id=block.id, tool_name=block.name))
                tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": result})

            api_messages.append({"role": "user", "content": tool_results})

# ─────────────────────────────────────────────
# Demos de cada versión
# ─────────────────────────────────────────────

SEP = "\n" + "─" * 50 + "\n"

async def demo_v1():
    print(SEP + "V1 · Agente básico" + SEP)
    agent = AgentV1(name="assistant", instructions="Eres un asistente conciso y útil. Responde en español.")
    response = await agent.run("¿Cuál es la capital de Francia y por qué es importante?")
    print(f"Agente: {response.final_content}")

async def demo_v2():
    print(SEP + "V2 · Agente con herramientas" + SEP)

    def get_weather(location: str) -> str:
        """Get current weather for a location."""
        data = {"Bilbao": "16°C, nublado", "Madrid": "24°C, soleado", "Barcelona": "21°C, parcialmente nublado"}
        return data.get(location, f"22°C, despejado en {location}")

    def calculate(expression: str) -> str:
        """Evaluate a safe math expression."""
        try:
            allowed = set("0123456789+-*/()., ")
            if not all(c in allowed for c in expression):
                return "Expresión no permitida"
            return f"{expression} = {eval(expression)}"
        except Exception as e:
            return f"Error: {e}"

    agent = AgentV2(
        name="assistant",
        instructions="Eres un asistente útil. Usa herramientas cuando sea necesario. Responde en español.",
        tools=[get_weather, calculate]
    )
    print("Pregunta: ¿Qué tiempo hace en Bilbao y cuánto es 15 * 24?\n")
    response = await agent.run("¿Qué tiempo hace en Bilbao y cuánto es 15 * 24?")
    print(f"\nAgente: {response.final_content}")

async def demo_v3():
    print(SEP + "V3 · Agente con memoria" + SEP)
    memory = ListMemory()
    agent = AgentV3(
        name="assistant",
        instructions="Eres un asistente útil con memoria. Responde en español.",
        memory=memory
    )
    print("Turno 1:")
    r1 = await agent.run("Me llamo Silvia y soy experta en AI en Iberdrola, en Bilbao.")
    print(f"Agente: {r1.final_content}\n")

    print("Turno 2 (el agente debería recordar el contexto):")
    r2 = await agent.run("¿Recuerdas quién soy y dónde trabajo?")
    print(f"Agente: {r2.final_content}")

async def demo_v4():
    print(SEP + "V4 · Streaming en tiempo real" + SEP)

    def get_time(city: str) -> str:
        """Get current local time for a city."""
        return f"14:32 CEST en {city}"

    agent = AgentV4(
        name="assistant",
        instructions="Eres un asistente experto en sistemas multiagente. Responde en español.",
        tools=[get_time]
    )

    print("Pregunta: ¿Qué son los sistemas multiagente?\n")
    print("Streaming → ", end="")

    async for event in agent.run_stream("Explícame en 3 puntos qué son los sistemas multiagente y por qué son útiles."):
        if isinstance(event, ToolCallEvent):
            print(f"\n  [tool_call] {event.tool_name}({event.parameters})")
        elif isinstance(event, ToolResultEvent):
            print(f"  [tool_result] {event.result}\n  → ", end="")
        elif isinstance(event, AgentResponse):
            print(f"\n\nRespuesta final completa ({len(event.final_content)} chars)")

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

async def main():
    arg = sys.argv[2] if len(sys.argv) > 2 and sys.argv[1] == "--v" else "all"

    if arg == "1" or arg == "all":
        await demo_v1()
    if arg == "2" or arg == "all":
        await demo_v2()
    if arg == "3" or arg == "all":
        await demo_v3()
    if arg == "4" or arg == "all":
        await demo_v4()

if __name__ == "__main__":
    asyncio.run(main())
