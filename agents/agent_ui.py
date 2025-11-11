import gradio as gr
import httpx
import json
import asyncio

# --- Gradio Interface ---

async def _stream_from_backend(query: str, queue: asyncio.Queue):
    """
    Handles the httpx streaming from the backend and puts events into a queue.
    """
    client = httpx.AsyncClient(timeout=180.0)
    try:
        async with client.stream("POST", "http://localhost:8000/api/agent/invoke",
                                 json={"query": query}, headers={"Accept": "text/event-stream"}) as response:
            if response.status_code >= 400:
                error_body = await response.aread()
                await queue.put({"type": "error", "content": f"Backend error: {response.status_code} - {error_body.decode()}"})
                await queue.put({"type": "stream_end"})
                return

            buffer = ""
            async for chunk in response.aiter_bytes():
                try:
                    decoded_chunk = chunk.decode("utf-8")
                    buffer += decoded_chunk
                    
                    # Process complete lines
                    while "\n\n" in buffer:
                        line, buffer = buffer.split("\n\n", 1)
                        line = line.strip()
                        
                        if line and line.startswith("data: "):
                            try:
                                json_string = line[len("data: "):]
                                json_data = json.loads(json_string)
                                await queue.put(json_data)
                            except json.JSONDecodeError as e:
                                print(f"Could not decode JSON: {line} - Error: {e}")
                                continue
                except UnicodeDecodeError as e:
                    print(f"Unicode decode error: {e}")
                    await queue.put({"type": "error", "content": f"Unicode decode error: {e}"})
                    continue
                    
    except httpx.TimeoutException:
        await queue.put({"type": "error", "content": "Request timed out. The agent might be taking too long to respond."})
    except httpx.RequestError as e:
        await queue.put({"type": "error", "content": f"Connection error: {str(e)}"})
    except Exception as e:
        await queue.put({"type": "error", "content": f"Unexpected error: {str(e)}"})
    finally:
        await client.aclose()
        # Don't add stream_end here, let the server handle it

async def stream_agent_response(query: str):
    """
    Streams responses from the agent backend, updating the UI in real-time.
    Matches the event types from agent_server_new_2.py:
    - thought
    - action
    - action_input
    - observation
    - final_answer_end
    - error
    - stream_end
    """
    if not query or not query.strip():
        yield "Please enter a valid query.", "", ""
        return

    # Initial yield to clear previous outputs and show loading
    yield "🔄 Starting agent...", "", ""

    queue = asyncio.Queue()
    stream_task = asyncio.create_task(_stream_from_backend(query.strip(), queue))

    thoughts_log = []
    context_log = []
    final_answer = ""
    timeout_counter = 0
    max_timeout = 300  # 5 minutes timeout
    received_events = False

    try:
        while True:
            try:
                # Wait for event with timeout
                event = await asyncio.wait_for(queue.get(), timeout=2.0)
                timeout_counter = 0
                received_events = True
                
                event_type = event.get("type")
                content = event.get("content", "")
                

                if event_type == "thought":
                    thoughts_log.append(f"💭 Thought: {content}")
                    thoughts_display = "\n\n".join(thoughts_log)
                    yield thoughts_display, final_answer, "\n\n".join(context_log)
                    await asyncio.sleep(0.1)  # Small delay for UI update
                    
                elif event_type == "action":
                    thoughts_log.append(f"⚡ Action: {content}")
                    thoughts_display = "\n\n".join(thoughts_log)
                    yield thoughts_display, final_answer, "\n\n".join(context_log)
                    await asyncio.sleep(0.1)
                    
                elif event_type == "action_input":
                    thoughts_log.append(f"📥 Action Input: {content}")
                    thoughts_display = "\n\n".join(thoughts_log)
                    yield thoughts_display, final_answer, "\n\n".join(context_log)
                    await asyncio.sleep(0.1)
                    
                elif event_type == "observation":
                    display_content = content 
                    context_log.append(f"🔍 Observation: {display_content}")
                    thoughts_display = "\n\n".join(thoughts_log)
                    context_display = "\n\n".join(context_log)
                    yield thoughts_display, final_answer, context_display
                    await asyncio.sleep(0.1)
                    
                elif event_type == "final_answer_end":
                    final_answer = content if content.strip() else "No response generated."
                    thoughts_display = "\n\n".join(thoughts_log)
                    context_display = "\n\n".join(context_log)
                    yield thoughts_display, final_answer, context_display
                    await asyncio.sleep(0.1)
                    
                elif event_type == "error":
                    error_msg = f"❌ Error: {content}"
                    thoughts_log.append(error_msg)
                    if not final_answer:
                        final_answer = error_msg
                    thoughts_display = "\n\n".join(thoughts_log)
                    yield thoughts_display, final_answer, "\n\n".join(context_log)
                    
                elif event_type == "stream_end":
                    if not final_answer:
                        # Check if we got any events
                        if thoughts_log or context_log:
                            final_answer = "Agent completed processing. Check the thoughts and observations above."
                        else:
                            final_answer = "Agent completed but no final answer was generated."
                    thoughts_display = "\n\n".join(thoughts_log)
                    context_display = "\n\n".join(context_log)
                    yield thoughts_display, final_answer, context_display
                    break
                    
            except asyncio.TimeoutError:
                timeout_counter += 1
                if timeout_counter >= max_timeout:
                    final_answer = (final_answer or "") + "\n\n⏱️ Response timed out."
                    yield "\n\n".join(thoughts_log), final_answer, "\n\n".join(context_log)
                    break
                # Continue waiting if we haven't received any events yet
                if not received_events:
                    continue
                # If we've been waiting too long after receiving events, end the stream
                if timeout_counter > 60:
                    if not final_answer:
                        final_answer = "Agent completed. Check the thoughts and observations."
                    yield "\n\n".join(thoughts_log), final_answer, "\n\n".join(context_log)
                    break
                continue
                
    except Exception as e:
        error_response = f"❌ UI Error: {str(e)}"
        thoughts_log.append(error_response)
        if not final_answer:
            final_answer = error_response
        yield "\n\n".join(thoughts_log), final_answer, "\n\n".join(context_log)
    finally:
        if not stream_task.done():
            stream_task.cancel()
            try:
                await stream_task
            except asyncio.CancelledError:
                pass

# Create the Gradio interface
agent_gradio_ui = gr.Interface(
    fn=stream_agent_response,
    inputs=gr.Textbox(
        lines=3, 
        label="Your Question / Instruction", 
        placeholder="e.g., What are the files in the 'rag' directory?",
        max_lines=5
    ),
    outputs=[
        gr.Textbox(
            lines=15,
            label="🧠 Agent Thoughts & Actions",
            show_copy_button=True
        ),
        gr.Textbox(
            lines=8,
            label="💬 Agent's Final Answer",
            show_copy_button=True
        ),
        gr.Textbox(
            lines=12,
            label="📄 Observations & Context",
            show_copy_button=True
        )
    ],
    title="🤖 Cogni-Code Agent",
    description="An intelligent ReAct agent that can analyze code, run commands, and interact with your codebase. Ask questions or give instructions!",
    allow_flagging="never",
    theme=gr.themes.Soft(),
    css="""
    .gradio-container {
        max-width: 1800px !important;
    }
    textarea {
        font-family: 'Monaco', 'Courier New', monospace;
        font-size: 14px;
    }
    """,
    examples=[
        "What files are in the current directory?",
        "Search for functions related to authentication",
        "What's the structure of this codebase?"
    ]
)
