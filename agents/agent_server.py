# agent_server.py – Real-time streaming with thread-safe capture

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json
import contextlib
import asyncio
import re
import sys
import queue as stdlib_queue
import threading

from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from agents.tools import list_files, read_file, run_shell_command, codebase_search
from clients.openai_client import get_llm

# ---------------------------------------------------------------------
router = APIRouter()

class InvokeRequest(BaseModel):
    query: str

# ---------------------------------------------------------------------
tools = [list_files, read_file, run_shell_command, codebase_search]

prompt_template = """
You are a ReAct agent. Your responses must follow the format: Thought, Action, Action Input.
Answer the following questions as best you can. You have access to the following tools:
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}
"""

prompt_template_new = """
You are a ReAct agent. Your responses must follow the format: Thought, Action, Action Input.
Answer the following questions as best you can. You have access to the following tools:
{tools}

Use the following format:

Question: the input question you must answer
Thought: You should always start by thinking about what to do. First, try to answer the question from your own knowledge. If you cannot, then consider the available tools.
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

**Tool Usage Strategy:**

1.  **Internal Knowledge:** Always try to answer directly from your existing knowledge first.
2.  **Basic File and Shell Tools:** If you need to interact with the file system or run commands, prefer `list_files`, `read_file`, or `run_shell_command`. These are for direct, specific tasks.
3.  **Codebase Search (Last Resort):** Only use `codebase_search` when you have a conceptual or exploratory question about the codebase and the other tools are not sufficient. Use it to find concepts or functionality when you don't know the specific file or location.

Begin!

Question: {input}
Thought:{agent_scratchpad}
"""

prompt = hub.pull("hwchase17/react")
prompt.template = prompt_template

_agent_executor = None
def get_agent_executor():
    global _agent_executor
    if _agent_executor is None:
        print("Initializing agent executor for the first time...")
        llm = get_llm()
        agent = create_react_agent(llm, tools, prompt=prompt)
        _agent_executor = AgentExecutor(agent=agent, tools=tools,
                                        verbose=True, handle_parsing_errors=True)
    return _agent_executor

# ---------------------------------------------------------------------
ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

class StreamCapture:
    """Capture stdout and yield lines in real-time."""
    def __init__(self, original, queue):
        self.original = original
        self.queue = queue
        self.buffer = ""
        
    def write(self, s):
        self.original.write(s)
        self.original.flush()
        self.buffer += s
        
        # Process complete lines
        while '\n' in self.buffer:
            line, self.buffer = self.buffer.split('\n', 1)
            self.queue.put(line)
        
        return len(s)
    
    def flush(self):
        self.original.flush()

async def get_agent_response_stream(query: str):
    """
    Stream agent stdout line-by-line in real-time.
    Tag lines starting with Thought:, Action:, etc.
    Begin after 'Entering new...' and stop after 'Finished chain.'.
    """
    print(f"Invoking agent with query: '{query}'")
    agent_executor = get_agent_executor()

    line_queue = stdlib_queue.Queue()
    stream_capture = StreamCapture(sys.stdout, line_queue)
    
    capturing = False
    current_tag = None
    current_lines = []

    def flush_current():
        if current_tag and current_lines:
            data = "\n".join(current_lines).strip()
            if data:
                clean = ansi_escape.sub('', data)
                return f"data: {json.dumps({'type': current_tag, 'content': clean})}\n\n"
        return None
    
    def process_line(line):
        """Process a single line and return message to yield, if any."""
        nonlocal capturing, current_tag, current_lines
        
        line = line.strip()
        if not line:
            return None

        # Clean escape codes
        line = ansi_escape.sub('', line)

        # Detect chain boundaries
        if "Entering new" in line and "chain" in line:
            capturing = True
            return None
        if "Finished chain." in line and capturing:
            # Flush any remaining buffer
            msg = flush_current()
            capturing = False
            return msg
        if not capturing:
            return None
        
        tool_indicators = ["Retrieved context from codebase search:",
                           "STDOUT", "STDERR"]

        # Detect tag transitions
        tag = None
        content = line
        if line.startswith("Thought:"):
            tag = "thought"
            content = line[len("Thought:"):].strip()
        elif line.startswith("Action:"):
            tag = "action"
            content = line[len("Action:"):].strip()
        elif line.startswith("Action Input:"):
            tag = "action_input"
            content = line[len("Action Input:"):].strip()
        elif line.startswith("Observation:"): 
            tag = "observation"
            content = line[len("Observation:"):].strip()
        elif line.startswith("STDOUT:"):
            tag="observation"
            content = line[len("STDOUT:"):].strip()
        elif line.startswith("STDERR:"):
            tag="observation"
            content = line[len("STDERR:"):].strip()
        elif line.startswith("CONTEXT:"):
            tag = "observation"
            content = line[len("CONTEXT:"):].strip()
        elif line.startswith("Final Answer:"):
            tag = "final_answer_end"
            content = line[len("Final Answer:"):].strip()

        if tag:
            # Flush previous block
            msg = flush_current()
            current_tag = tag
            current_lines = [content] if content else []
            return msg
        else:
            # Continuation of current block
             current_lines.append(content)
             return None

    try:
        # Start agent in a separate thread
        agent_result = {}
        agent_error = {}
        
        def run_agent():
            try:
                with contextlib.redirect_stdout(stream_capture):
                    result = agent_executor.invoke({"input": query})
                    agent_result['data'] = result
            except Exception as e:
                agent_error['error'] = e
        
        agent_thread = threading.Thread(target=run_agent)
        agent_thread.start()
        
        # Process lines as they come in
        while agent_thread.is_alive() or not line_queue.empty():
            try:
                line = line_queue.get(timeout=0.1)
                msg = process_line(line)
                if msg:
                    yield msg
                    await asyncio.sleep(0.05)
            except stdlib_queue.Empty:
                await asyncio.sleep(0.05)
                continue
        
        # Process any remaining lines
        while not line_queue.empty():
            try:
                line = line_queue.get_nowait()
                msg = process_line(line)
                if msg:
                    yield msg
            except stdlib_queue.Empty:
                break
        
        # Wait for thread to complete
        agent_thread.join(timeout=5)
        
        # Check for errors
        if 'error' in agent_error:
            raise agent_error['error']

        # Flush last block
        msg = flush_current()
        if msg:
            yield msg

        yield f"data: {json.dumps({'type': 'stream_end'})}\n\n"

    except Exception as e:
        print(f"[ERROR] Agent execution failed: {e}")
        yield f"data: {json.dumps({'type': 'error','content': f'Agent execution failed: {e}'})}\n\n"
        yield f"data: {json.dumps({'type': 'stream_end'})}\n\n"

# ---------------------------------------------------------------------
@router.post("/agent/invoke")
async def agent_invoke(request: InvokeRequest):
    return StreamingResponse(
        get_agent_response_stream(request.query),
        media_type="text/event-stream"
    )
