"""
xmad/1_iteration/tests/test_tool_calling.py

Isolated check that the LLama vLLM endpoint actually emits native tool calls.
No MCP, no agent loop -- this only confirms the model + parser are configured
correctly, BEFORE we wire tool calling into the full agent loop.

Prerequisite: serve Llama2-7B WITH tool calling enabled:
    vllm serve <path>/Llama2-7B-Instruct \
        --served-model-name llama --port 7556 \
        --enable-auto-tool-choice --tool-call-parser hermes

Then:
    pip install openai
    python tests/test_tool_calling.py

Expect: a non-empty tool_calls list naming search_knowledge_base.
"""

from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:7556/v1", api_key="not-needed")

# The same tool schema the agent will advertise later (here hand-written; in the
# loop we will generate it from the MCP server's tool list).
tools = [{
    "type": "function",
    "function": {
        "name": "search_knowledge_base",
        "description": "Search the knowledge base for passages relevant to the query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "A natural-language search query."},
                "k": {"type": "integer", "description": "How many passages to return."},
            },
            "required": ["query"],
        },
    },
}]

resp = client.chat.completions.create(
    model="llama",
    messages=[
        {"role": "system",
         "content": "You are a research assistant. Use the search_knowledge_base "
                    "tool to look things up before answering."},
        {"role": "user",
         "content": "What is the nationality of the director Scott Derrickson? Look it up."},
    ],
    tools=tools,
    tool_choice="auto",
)

msg = resp.choices[0].message
print("finish_reason:", resp.choices[0].finish_reason)
print("content:", msg.content)
print("tool_calls:", msg.tool_calls)

if msg.tool_calls:
    print(f"\nPASS: model emitted {len(msg.tool_calls)} tool call(s)")
    for tc in msg.tool_calls:
        print(f"  -> {tc.function.name}({tc.function.arguments})")
else:
    print("\nFAIL: no tool_calls. Check --enable-auto-tool-choice and "
          "--tool-call-parser hermes on the vLLM server.")
