# CLI Coding Assistant

An autonomous coding agent powered by **LiteLLM** supporting **OpenAI, Anthropic, Google Gemini, Mistral, Ollama**, and 100+ other LLM providers. A single intelligent agent works directly in your terminal: it plans, writes code, tests, and self-corrects, all in one loop. For complex tasks, it can spawn controlled sub-agents.

![Python Version](https://img.shields.io/badge/python-≥3.12-blue.svg)
![LiteLLM](https://img.shields.io/badge/Powered%20By-LiteLLM-blueviolet.svg)

---

## Demo
<!-- Add your demo video link here, e.g. YouTube or mp4 URL -->
[Watch the Demo Video](https://github.com/user-attachments/assets/8bae0486-4366-491f-98e6-e08766f5c7a2)

---

## Key Features

*   **Single-Agent Architecture (Inspired by Claude Code)**:
    *   One main agent that plans inline, codes, and verifies — no handoff overhead.
    *   Can spawn **controlled sub-agents** for complex, isolated subtasks (sub-agents cannot nest further).
*   **ReAct Loop**: The agent follows a **Think → Act → Verify → Fix** cycle. It reasons about the task, calls tools, checks results, and self-corrects until the task is done — then returns control to the user.
*   **Robust Autonomous Tooling**:
    *   **Smart File Editing**: Preferred `edit_file` tool uses exact-match first, then fuzzy `difflib` fallback — avoiding full-file rewrites for small changes.
    *   **Syntax Checking**: `py_compile`-based compiler catches syntax errors before execution.
    *   **Fast Package Management**: Uses `uv add` to install missing dependencies on the fly.
    *   **Web Search**: DuckDuckGo integration via `ddgs` for real-time documentation lookup.
    *   **Project Tracking**: Automatic `PROGRESS.md` generation with checklists to track milestones.
*   **Multi-Tiered Safety**: Interactive approval prompts for writes, edits, deletes, and execution — with an **Approve All** shortcut for hands-off mode.
*   **Context-Aware Memory Management**:
    *   Monitors conversation length, auto-summarizes old messages via LLM when nearing context limits.
    *   Preserves recent tool calls in full while compressing older history.
    *   Sub-agents run in isolated contexts, preventing memory pollution.
*   **Multi-Provider Support**: Automatic API key resolution — just pass a `--model` flag and the right key is picked from your `.env`.
*   **Beautiful Terminal UI**: Uses the `rich` library for markdown rendering, colored diffs, spinners, and styled panels.

---

## Architecture
<img width="1503" height="1215" alt="image" src="https://github.com/user-attachments/assets/17f81c31-4892-4d1b-b14d-37def3242fff" />


---


## Project Structure

```text
coder-agent/
│
├──  main.py               # CLI entry point — arg parsing, API key resolution, REPL loop
├──  agent.py              # Main ReAct agent loop, system prompt, file-tree injection
├──  agent_tools.py        # Tool schemas for main agent (AGENT_TOOLS) and sub-agent (SUBAGENT_TOOLS)
├──  agent_helpers.py      # Tool execution, approval flow, diff display, memory trimming + summarization
├──  subagent.py           # Isolated sub-agent — own ReAct loop, auto-nudge, finish_task exit
├──  ai_utils.py           # LiteLLM completion wrapper with tenacity retry (rate limits, timeouts)
│
├── functions/             # Tool implementations (sandboxed to working directory)
│   ├── get_files_info.py  # Recursive directory tree with smart ignore list
│   ├── get_file_content.py# Read file with 10k char truncation guard
│   ├── write_file.py      # Create / overwrite file (requires parent dir to exist)
│   ├── edit_file.py       # Search/replace with exact → fuzzy fallback via difflib
│   ├── delete_file.py     # Safe file deletion with path validation
│   ├── create_directory.py# Recursive mkdir with idempotency
│   ├── run_compiler.py    # py_compile syntax checker
│   ├── run_python_file.py # Subprocess execution with 30s timeout + output truncation
│   ├── web_search.py      # DuckDuckGo search via ddgs
│   ├── install_package.py # uv add package installer
│   └── project_state.py   # PROGRESS.md read/write for milestone tracking
│
├── docs/                  # Module-level documentation
└── workspace/             # Default directory where the agent builds your projects
```

---

## Getting Started

### Prerequisites

1.  **Python ≥ 3.12** installed.
2.  **An API Key** from at least one supported provider (e.g. [Mistral AI](https://mistral.ai/), [OpenAI](https://platform.openai.com/), [Anthropic](https://console.anthropic.com/), [Google Gemini](https://aistudio.google.com/)).
3.  **[uv](https://docs.astral.sh/uv/)** — used by the agent's `install_package` tool to manage dependencies.

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/sachinmehan3/CLI-Coding-Assistant.git
    cd CLI-Coding-Assistant
    ```

2.  Install dependencies (using uv or pip):
    ```bash
    # With uv (recommended)
    uv sync

    # Or with pip
    pip install litellm rich python-dotenv ddgs tenacity
    ```

3.  Create a `.env` file with the key for your chosen provider:
    ```env
    # Only the key matching your --model flag is required
    MISTRAL_API_KEY=your_mistral_key_here
    # OPENAI_API_KEY=sk-your-openai-key-here
    # ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here
    # GEMINI_API_KEY=your-gemini-key-here
    ```

---

## Usage

Start by running `main.py`. You can optionally specify a target directory and/or an LLM model.

```bash
# Default (Mistral Medium)
python main.py --dir my_new_project

# Use OpenAI
python main.py --model gpt-4o

# Use Anthropic
python main.py --model anthropic/claude-sonnet-4-20250514

# Use Google Gemini
python main.py --model gemini/gemini-2.0-flash

# Use local Ollama
python main.py --model ollama/llama3
```

### In-Session Commands

| Command  | Description                        |
|----------|------------------------------------|
| `/clear` | Clears the terminal screen         |
| `exit`   | Exits the session                  |

During execution the agent will prompt `(y)es / (n)o / (a)pprove all` before any destructive action (write, edit, delete, execute). Choose **a** to enable auto-approval for the rest of the current task.

---

## How It Works

1.  **User sends a message** → appended to conversation history.
2.  **File tree injected** into the system prompt so the agent always knows the project layout.
3.  **LLM generates** a response — either pure text (returned to user) or tool calls.
4.  **Tools execute** through `agent_helpers.py`, which routes each call to the right `functions/` implementation, shows diffs, and prompts for approval.
5.  **Tool results** are appended to history and the loop continues until the agent responds with text only.
6.  **Memory management** kicks in when conversation history exceeds ~120k characters — older messages are LLM-summarized while recent context is preserved in full.
7.  **Sub-agents** (via `spawn_subagent`) run the same loop in an isolated context and return a summary to the main agent upon calling `finish_task`.

---

## Memory Management

*   **Active Summarization**: When history grows past the limit, `memory.py` compresses older messages into a dense LLM-generated summary, preserving key decisions without context bloat.
*   **Sliding Window**: The most recent 8 messages (plus all system prompts) are always preserved in full.
*   **Summary Stacking**: Previously generated summaries are preserved verbatim — only new unsummarized messages are compressed, preventing information loss over long sessions.
*   **Sub-Agent Isolation**: Sub-agents get their own context and memory, preventing complex subtasks from polluting the main agent's history.

---

## Safety & Sandboxing

*   **Path Validation**: Every file/directory tool validates that the target path stays within the working directory — no escaping to system files.
*   **Approval Prompts**: Writes, edits, deletes, and script execution all require explicit user approval (or auto-approve mode).
*   **Execution Timeout**: `run_python_file` enforces a 30-second timeout with `stdin=DEVNULL` to prevent infinite loops and hanging prompts.
*   **Output Truncation**: Script output is capped at 50 lines to avoid blowing up the agent's context window.
*   **File Read Limits**: `get_file_content` truncates at 10,000 characters.
*   **No GUI Execution**: The system prompt and tool descriptions explicitly forbid running GUI apps or blocking servers.

---

## Future Additions

*   **Docker Containerization**: Isolate the agent's environment and workspace for enhanced security, reproducibility, and easier deployment.
*   **Streaming & Async**: Implement asynchronous operations and streamed responses for a faster, more responsive terminal experience while the agent generates code and plans.
*   **Multi-File Editing**: Batch edits across multiple files in a single tool call for large refactors.

---

## 🛡️ License
This project is open-source and available under the MIT License.
