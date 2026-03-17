import json
import time
from rich.markdown import Markdown

from functions.get_files_info import get_file_info
from ai_utils import safe_completion
from agent_tools import SUBAGENT_TOOLS
from agent_helpers import trim_memory, execute_tool
from token_tracker import get_max_context_tokens

SUBAGENT_SYSTEM_PROMPT = (
    "You are an autonomous Sub-Agent spawned by a main Agent to handle a specific task. "
    "Complete the task fully and correctly using your tools, then call `finish_task` with a clear summary.\n\n"

    "AVAILABLE TOOLS & WHEN TO USE THEM:\n"
    "- `get_files_info`: Map out the directory structure and discover files.\n"
    "- `get_file_content`: Read a file BEFORE modifying it to gather its exact contents.\n"
    "- `write_file`: Create a NEW file or modify an EXISTING file. You MUST provide the ENTIRE, complete file content. Use ONLY for new files or full rewrites.\n"
    "- `edit_file`: Edit an existing file by search/replace. PREFERRED for modifications — provide the exact text to find and its replacement.\n"
    "- `delete_file`: Delete an existing file.\n"
    "- `create_directory`: Create nested folders BEFORE writing files into them.\n"
    "- `run_compiler`: Check `.py` files for syntax errors. ALWAYS do this BEFORE running new code.\n"
    "- `run_python_file`: Execute a script for STDOUT/STDERR. NEVER execute GUI apps or blocking servers.\n"
    "- `web_search`: Look up documentation, APIs, or tutorials.\n"
    "- `install_package`: Install PyPI dependencies when hitting ModuleNotFoundError.\n"
    "- `ask_user`: Ask the user for help if you are repeatedly failing or need clarification.\n"
    "- `finish_task`: End your turn. Provide a clear summary of what you did.\n\n"

    "RULES:\n"
    "1. If a file exists, `get_file_content` it first before overwriting.\n"
    "2. All paths are relative. Do not invent absolute paths.\n"
    "3. Be brief. State your next action in 1-2 sentences before calling a tool.\n"
    "4. If `run_compiler` or `run_python_file` yields errors, FIX them before calling `finish_task`.\n"
    "5. You are a text-only bot. NEVER open, execute, or interact with GUI files or windows.\n"
)


def run_subagent(model, console, task_description, working_dir, tracker=None):
    """Runs an isolated sub-agent that completes a task and returns a summary."""

    tools = SUBAGENT_TOOLS
    approve_all = [False]

    messages = [
        {"role": "system", "content": SUBAGENT_SYSTEM_PROMPT},
        {"role": "user", "content": task_description},
    ]

    # Inject file tree once as a standalone message — never mutate messages[0].
    file_tree = get_file_info(working_dir, ".")
    messages.append({"role": "system", "content": f"CURRENT PROJECT FILES:\n{file_tree}"})

    console.print(f"\n[bold magenta] Sub-Agent spawned[/bold magenta]")

    MAX_ITERATIONS = 200
    iteration = 0

    while iteration < MAX_ITERATIONS:
        iteration += 1

        max_tokens = get_max_context_tokens(model)
        messages = trim_memory(messages, max_tokens, console, model)

        with console.status("[bold magenta]Sub-Agent thinking...[/bold magenta]", spinner="dots"):
            t0 = time.time()
            response = safe_completion(
                model=model,
                messages=messages,
                tools=tools
            )
            latency_ms = (time.time() - t0) * 1000
            
            if tracker:
                tracker.record(response)
                
            assistant_message = response.choices[0].message
            full_content = assistant_message.content if assistant_message.content else ""
            
            stitched_tools = {}
            if assistant_message.tool_calls:
                for idx, tool_call in enumerate(assistant_message.tool_calls):
                    stitched_tools[idx] = {
                        "id": tool_call.id,
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    }

        # Show the sub-agent's reasoning — styled like the main agent
        if full_content.strip():
            if stitched_tools:
                # Mid-loop reasoning (tool calls coming) — dim style
                console.print(f"[dim magenta]  {full_content.strip()}[/dim magenta]")
            else:
                # Standalone text output — full Markdown
                console.print(Markdown(full_content))

        assistant_msg = {
            "role": "assistant",
            "content": full_content
        }
        
        parsed_tool_calls = []
        
        if stitched_tools:
            tool_calls_list = []
            for idx, tc in stitched_tools.items():
                parsed_tool_calls.append(tc)
                tool_calls_list.append({
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": tc["arguments"]
                    }
                })
            assistant_msg["tool_calls"] = tool_calls_list
            
        messages.append(assistant_msg)
        
        if parsed_tool_calls:
            for tc in parsed_tool_calls:
                function_name = tc["name"]
                args_string = tc["arguments"]
                tool_call_id = tc["id"]
                
                try:
                    args = json.loads(args_string)
                except json.JSONDecodeError:
                    args = {}

                if function_name == "finish_task":
                    summary = args.get("summary", "Sub-agent completed without summary.")
                    console.print(f"[bold magenta] Sub-Agent finished[/bold magenta]")
                    return summary
                
                function_result = execute_tool(function_name, args, working_dir, approve_all, console)


                messages.append({
                    "role": "tool",
                    "name": function_name,
                    "content": str(function_result),
                    "tool_call_id": tool_call_id 
                })
            continue 
        else:
            nudge = "SYSTEM: You output text but did not call a tool. If the task is finished, call `finish_task`. Otherwise, use the appropriate tool to continue."
            messages.append({"role": "user", "content": nudge})
            continue
