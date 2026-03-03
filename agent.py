import json
from rich.markdown import Markdown
from rich.panel import Panel

from functions.get_files_info import get_file_info
from ai_utils import safe_completion
from agent_tools import AGENT_TOOLS
from agent_helpers import trim_memory, execute_tool
from subagent import run_subagent

AGENT_SYSTEM_PROMPT = (
    "You are an expert, fully autonomous coding agent working inside a project directory. "
    "You talk directly to the user, plan your approach, and execute everything yourself using your tools.\n\n"

    "HOW TO WORK:\n"
    "0. TRACK PROGRESS: At the START of every new project or multi-step task, call `update_tracker` "
    "to create PROGRESS.md with the goal and milestones. Update it after completing each milestone. "
    "When you are FINISHED, call `update_tracker` one final time to mark ALL completed items as `[x]` "
    "and set the status to 'Complete'. If you built something but cannot test it yourself "
    "(e.g. a GUI app), still mark it as done — do NOT leave items unchecked just because you "
    "cannot self-verify.\n"
    "1. When the user gives you a task, THINK first — briefly state your plan in 1-3 sentences.\n"
    "2. Then ACT — call the appropriate tools to implement your plan.\n"
    "3. VERIFY — compile and/or run your code to confirm it works.\n"
    "4. If something fails, FIX it and try again.\n"
    "5. When done, respond to the user with a clear summary of what you built or changed.\n\n"

    "AVAILABLE TOOLS:\n"
    "- `get_files_info`: Map out the directory structure.\n"
    "- `get_file_content`: Read a file's contents. ALWAYS do this before modifying an existing file.\n"
    "- `write_file`: Create or overwrite a file. Provide the ENTIRE file content. Use ONLY for new files or full rewrites.\n"
    "- `edit_file`: Edit an existing file by search/replace. PREFERRED for modifications — provide the exact text block to find and its replacement.\n"
    "- `delete_file`: Delete a file.\n"
    "- `create_directory`: Create directories before writing files into them.\n"
    "- `run_compiler`: Syntax-check a `.py` file without executing it.\n"
    "- `run_python_file`: Execute a script. NEVER run GUI apps or blocking servers.\n"
    "- `web_search`: Search the web for docs or tutorials.\n"
    "- `install_package`: Install a PyPI package.\n"
    "- `ask_user`: Ask the user a question if you need clarification or are stuck.\n"
    "- `spawn_subagent`: Delegate a complex, self-contained subtask to a sub-agent. "
    "The sub-agent runs in its own context and returns a summary. Use this for large tasks "
    "where you want to keep your own context clean.\n"
    "- `update_tracker`: Create or update the project progress tracker (PROGRESS.md). "
    "Write the full markdown content with checklists to track milestones.\n\n"

    "RULES:\n"
    "1. NO BLIND OVERWRITES: If a file exists, `get_file_content` it first.\n"
    "2. RELATIVE PATHS ONLY: All paths are relative to the project root.\n"
    "3. BREVITY: State your plan concisely before acting. No essays.\n"
    "4. SELF-CORRECTION: If compilation or execution fails, fix the errors yourself.\n"
    "5. NO VISUAL GUIs: You are text-only. Never open, execute, or interact with GUI windows.\n"
    "6. CONTEXT: The current project file tree is injected into your system prompt automatically.\n"
)


def get_initial_messages():
    """Returns the initial message list for a fresh agent session."""
    return [{"role": "system", "content": AGENT_SYSTEM_PROMPT}]


def run_agent_loop(model, console, working_dir, user_input, messages):
    """
    Single ReAct loop. The agent processes user input, calls tools as needed,
    and returns control to the user when it produces a text-only response.
    Returns the updated messages list.
    """
    messages.append({"role": "user", "content": user_input})
    approve_all = [False]
    file_tree_cache = None
    tree_dirty = True
    FILE_MODIFYING_TOOLS = {"write_file", "edit_file", "delete_file", "create_directory"}

    MAX_ITERATIONS = 200
    iteration = 0

    while iteration < MAX_ITERATIONS:
        iteration += 1
        try:
            MAX_TOKENS = 30000
            messages = trim_memory(messages, MAX_TOKENS, console, model)

            # Refresh file tree only when dirty
            if tree_dirty:
                file_tree_cache = get_file_info(working_dir, ".")
                tree_dirty = False
            messages[0]["content"] = (
                f"{AGENT_SYSTEM_PROMPT}\n\n"
                f" CURRENT PROJECT FILES:\n{file_tree_cache}\n\n"
            )

            with console.status("[bold cyan]Thinking...[/bold cyan]", spinner="dots"):
                response = safe_completion(
                    model=model,
                    messages=messages,
                    tools=AGENT_TOOLS
                )
                
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

            # Build the assistant message for chat history
            assistant_msg = {"role": "assistant", "content": full_content}
            
            # Show reasoning — styled differently depending on whether tool calls follow
            if full_content.strip():
                if stitched_tools:
                    # Mid-loop reasoning (tool calls coming) — dim style
                    console.print(f"[dim cyan]  {full_content.strip()}[/dim cyan]")
                else:
                    # Final response (no more tools) — full Markdown
                    console.print(Markdown(full_content))

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

            # If there are tool calls, execute them and loop
            if parsed_tool_calls:
                for tc in parsed_tool_calls:
                    function_name = tc["name"]
                    args_string = tc["arguments"]
                    tool_call_id = tc["id"]
                    
                    try:
                        args = json.loads(args_string)
                    except json.JSONDecodeError:
                        args = {}

                    # Handle spawn_subagent specially
                    if function_name == "spawn_subagent":
                        task_desc = args.get("task_description", "")
                        subagent_result = run_subagent(model, console, task_desc, working_dir)
                        
                        messages.append({
                            "role": "tool",
                            "name": "spawn_subagent",
                            "content": f"SUB-AGENT RESULT:\n{subagent_result}",
                            "tool_call_id": tool_call_id
                        })
                        tree_dirty = True
                        continue

                    function_result = execute_tool(function_name, args, working_dir, approve_all, console)

                    if function_name in FILE_MODIFYING_TOOLS:
                        tree_dirty = True

                    messages.append({
                        "role": "tool",
                        "name": function_name,
                        "content": str(function_result),
                        "tool_call_id": tool_call_id
                    })
                continue

            # No tool calls, agent produced a text response. Return control to user.
            else:
                return messages
                
        except Exception as e:
            import traceback
            console.print(f"[bold red]Error in Agent Loop:[/bold red] {e}")
            console.print(f"[bold red]Traceback:[/bold red]\n{traceback.format_exc()}")
            return messages
