import os
import difflib
import litellm
from rich.markdown import Markdown
from rich.panel import Panel


from functions.get_files_info import get_file_info
from functions.get_file_content import get_file_content
from functions.write_file import write_file
from functions.edit_file import edit_file
from functions.delete_file import delete_file
from functions.create_directory import create_directory
from functions.run_python_file import run_python_file
from functions.web_search import web_search
from functions.install_package import install_package
from functions.run_compiler import run_compiler
from functions.project_state import get_progress, write_progress

from ai_utils import safe_completion
from logger import get_logger
from exceptions import ToolExecutionError, ToolNotFoundError


def summarize_history(model, messages_to_summarize):
    """Compresses older conversation history into a dense LLM-generated summary."""
    conversation_text = ""
    for msg in messages_to_summarize:
        raw_role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", "")
        raw_content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")
        raw_name = msg.get("name") if isinstance(msg, dict) else getattr(msg, "name", "")
        
        role = str(raw_role) if raw_role is not None else ""
        content = str(raw_content) if raw_content is not None else ""
        name = str(raw_name) if raw_name is not None else ""
        
        tool_calls = msg.get("tool_calls", []) if isinstance(msg, dict) else getattr(msg, "tool_calls", [])
        if tool_calls:
            for tc in tool_calls:
                func = tc.get("function", {}) if isinstance(tc, dict) else getattr(tc, "function", None)
                func_name = func.get("name", "") if isinstance(func, dict) else getattr(func, "name", "")
                func_args = func.get("arguments", "") if isinstance(func, dict) else getattr(func, "arguments", "")
                content = content + f"\n[ACTION TAKEN: Called tool '{func_name}' with instructions: {func_args}]"

        prefix = f"{role} ({name})" if name else role
        safe_content = str(content)[:2000] + ("..." if len(str(content)) > 2000 else "")
        conversation_text += f"[{prefix.upper()}]: {safe_content}\n"

    prompt = (
        "You are the agent's memory module. Summarize the following conversation history. "
        "Focus strictly on: 1) What tasks have been completed. 2) What decisions were made. "
        "3) The current state of the codebase. "
        "Be highly concise, technical, and accurate. Do not add fluff.\n\n"
        f"HISTORY TO SUMMARIZE:\n{conversation_text}"
    )

    response = safe_completion(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


# Tool results larger than this (in chars) will be shrunk once they leave the recent window
SHRINK_THRESHOLD = 500
PROTECT_RECENT = 8


def shrink_old_tool_results(messages, protect_recent=PROTECT_RECENT):
    """Replace large, already-processed tool results with compact summaries.
    
    Keeps recent messages intact (they may still be needed for the current
    reasoning chain) but shrinks older tool outputs that the agent has 
    already acted upon — preventing stale file reads and command outputs
    from wasting context for the rest of the session.
    """
    if len(messages) <= protect_recent + 1:  # +1 for system prompt
        return messages
    
    cutoff = len(messages) - protect_recent
    
    for i in range(1, cutoff):  # Skip system prompt at index 0
        msg = messages[i]
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "tool":
            continue
            
        content = str(msg.get("content", ""))
        if len(content) <= SHRINK_THRESHOLD:
            continue
        
        # Already shrunk in a previous pass
        if content.startswith("[Shrunk tool result"):
            continue
        
        tool_name = msg.get("name", "unknown")
        char_count = len(content)
        preview = content[:200].rstrip()
        
        msg["content"] = (
            f"[Shrunk tool result from '{tool_name}' — originally {char_count:,} chars]\n"
            f"Preview: {preview}..."
        )
    
    return messages


def trim_memory(messages, max_tokens, console, model):
    """Trims the agent's memory to stay within context window limits using token counting."""
    # Proactively shrink old tool results before counting tokens
    messages = shrink_old_tool_results(messages)

    def count_message_tokens(msg):
        """Count tokens for a single message using the model's tokenizer."""
        content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
        tool_calls = msg.get("tool_calls", []) if isinstance(msg, dict) else getattr(msg, "tool_calls", [])
        text = str(content or "")
        if tool_calls:
            text += " " + str(tool_calls)
        try:
            return litellm.token_counter(model=model, text=text)
        except Exception:
            return len(text) // 4  # Fallback: ~4 chars per token

    total_tokens = sum(count_message_tokens(m) for m in messages)

    if total_tokens > max_tokens:
        console.print(f"\n[dim]Memory reached {total_tokens:,} tokens (limit: {max_tokens:,}). Summarizing older messages...[/dim]")
        
        system_prompt = messages[0]
        tail = messages[-8:]
        
        # Drop orphaned 'tool' messages at the start of the tail
        while len(tail) > 0:
            first_msg = tail[0]
            role = first_msg.get("role") if isinstance(first_msg, dict) else getattr(first_msg, "role", "")
            if role == "tool":
                tail.pop(0)
            else:
                break
                
        middle_messages = messages[1 : len(messages) - len(tail)]
        
        # Preserve existing summaries verbatim instead of re-summarizing them
        old_summaries = []
        regular_messages = []
        for msg in middle_messages:
            content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
            if str(content).startswith("PREVIOUS CONVERSATION SUMMARY:"):
                old_summaries.append(str(content))
            else:
                regular_messages.append(msg)
        
        new_summary = summarize_history(model, regular_messages) if regular_messages else ""
        
        # Combine: old summaries preserved as-is, new summary appended
        combined_parts = old_summaries + ([new_summary] if new_summary else [])
        combined_text = "\n\n".join(combined_parts)
        
        summary_message = {
            "role": "system", 
            "content": f"PREVIOUS CONVERSATION SUMMARY:\n{combined_text}"
        }
        
        messages = [system_prompt, summary_message] + tail
        after_tokens = sum(count_message_tokens(m) for m in messages)
        get_logger().log_memory_trim(before_tokens=total_tokens, after_tokens=after_tokens)
        console.print(f"[dim]Memory optimized. Resuming with {after_tokens:,} tokens.[/dim]")

    return messages


def show_diff(console, old_text, new_text, file_path):
    """Prints a colored unified diff to the console."""
    old_lines = old_text.splitlines(keepends=True)
    new_lines = new_text.splitlines(keepends=True)
    diff = difflib.unified_diff(old_lines, new_lines, fromfile=f"a/{file_path}", tofile=f"b/{file_path}")

    has_diff = False
    for line in diff:
        has_diff = True
        line = line.rstrip("\n")
        if line.startswith("+") and not line.startswith("+++"):
            console.print(f"{line}")
        elif line.startswith("-") and not line.startswith("---"):
            console.print(f"{line}")
        elif line.startswith("@@"):
            console.print(f"{line}")
        else:
            console.print(f"[dim]{line}[/dim]")

    if not has_diff:
        console.print("[dim]No changes detected.[/dim]")


def ask_approval(console, message, approve_all):
    """Prompts for approval. Returns True if approved. Handles 'a' to enable approve-all."""
    if approve_all[0]:
        console.print(f"[dim]Auto-approved: {message}[/dim]")
        return True
    
    console.print(f"\n[bold]Authorization Required: {message}[/bold]")
    approval = ''
    while approval not in ['y', 'yes', 'n', 'no', 'a']:
        approval = console.input("[bold](y)es / (n)o / (a)pprove all > [/bold]").strip().lower()
    
    if approval == 'a':
        approve_all[0] = True
        console.print("[bold]Approve-all enabled for this task.[/bold]")
        return True
    
    return approval in ['y', 'yes']


def execute_tool(function_name, args, working_dir, approve_all, console):
    """Executes a single tool and returns the result string.
    
    Handles errors gracefully — tool failures return clear error messages
    instead of crashing the session.
    """
    function_result = ""

    try:
        if function_name in ["get_files_info", "get_file_content", "create_directory", "web_search", "run_compiler"]:
            with console.status(f"[bold]Executing {function_name}...[/bold]", spinner="dots"):
                if function_name == "get_files_info":
                    function_result = get_file_info(working_dir, args.get("directory", "."))
                    console.print(f"[dim]Checked directory tree[/dim]")
                    
                elif function_name == "get_file_content":
                    function_result = get_file_content(working_dir, args.get("file_path"))
                    console.print(f"[dim]Read file: {args.get('file_path')}[/dim]")
                    
                elif function_name == "create_directory":
                    function_result = create_directory(working_dir, args.get("directory_path"))
                    console.print(f"[dim]Created directory: {args.get('directory_path')}[/dim]")
                    
                elif function_name == "web_search":
                    try:
                        function_result = web_search(args.get("query"))
                        console.print(f"[dim]Searched web for: {args.get('query')}[/dim]")
                    except Exception as e:
                        get_logger().log_error("web_search", e)
                        function_result = f"Error: Web search is temporarily unavailable ({type(e).__name__}: {e}). Try proceeding without search or ask the user for help."
                        console.print(f"[dim yellow]Web search failed (graceful degradation)[/dim yellow]")
                    
                elif function_name == "run_compiler":
                    function_result = run_compiler(working_dir, args.get("file_path"))
                    if "FATAL SYNTAX ERROR" in function_result or "Error" in function_result:
                        console.print(Panel(function_result, title=f"Compile Failed: {args.get('file_path')}"))
                    else:
                        console.print(f"[bold]Success:[/bold] {function_result}")

        elif function_name == "write_file":
            file_path = args.get("file_path")
            content = args.get("content")
            
            if not approve_all[0]:
                abs_path = os.path.join(os.path.abspath(working_dir), file_path)
                if os.path.isfile(abs_path):
                    with open(abs_path, "r", encoding="utf-8") as f:
                        old_content = f.read()
                    show_diff(console, old_content, content, file_path)
                else:
                    console.print(f"[dim](new file — {len(content)} chars)[/dim]")
            
            if ask_approval(console, f"Agent wants to write '{file_path}'", approve_all):
                with console.status(f"[bold]Writing {file_path}...[/bold]", spinner="dots"):
                    function_result = write_file(working_dir, file_path, content)
                    console.print(f"[dim]Wrote file: {file_path}[/dim]")
            else:
                function_result = "SYSTEM ERROR: User denied permission to write file."

        elif function_name == "edit_file":
            file_path = args.get("file_path")
            search = args.get("search", "")
            replace = args.get("replace", "")
            
            if not approve_all[0]:
                show_diff(console, search, replace, file_path)
            
            if ask_approval(console, f"Agent wants to edit '{file_path}'", approve_all):
                with console.status(f"[bold]Editing {file_path}...[/bold]", spinner="dots"):
                    function_result = edit_file(working_dir, file_path, search, replace)
                    console.print(f"[dim]Edited file: {file_path}[/dim]")
            else:
                function_result = "SYSTEM ERROR: User denied permission to edit file." 

        elif function_name == "delete_file":
            file_path = args.get("file_path")
            
            if ask_approval(console, f"Agent wants to delete '{file_path}'", approve_all):
                with console.status(f"[bold]Deleting {file_path}...[/bold]", spinner="dots"):
                    function_result = delete_file(working_dir, file_path)
                    console.print(f"[dim]Deleted file: {file_path}[/dim]")
            else:
                function_result = "SYSTEM ERROR: User denied permission to delete file."

        elif function_name == "run_python_file":
            file_path = args.get("file_path")
            script_args = args.get("args", [])
            
            if ask_approval(console, f"Agent wants to execute '{file_path}'", approve_all):
                with console.status(f"[bold]Executing {file_path}...[/bold]", spinner="dots"):
                    function_result = run_python_file(working_dir, file_path, script_args)
                # Show execution output to the user in a visible panel
                output_text = function_result.strip()
                if "Error" in function_result or "Traceback" in function_result or "Process exited with code" in function_result:
                    console.print(Panel(output_text, title=f"Execution Failed: {file_path}"))
                else:
                    console.print(Panel(output_text, title=f"Execution Output: {file_path}"))
            else:
                function_result = "SYSTEM ERROR: User denied permission."
        
        elif function_name == "install_package":
            package_name = args.get("package_name")
            
            if ask_approval(console, f"Agent wants to install package: '{package_name}'", approve_all):
                with console.status(f"[bold]Installing {package_name}...[/bold]", spinner="dots"):
                    function_result = install_package(working_dir, package_name)
                    console.print(f"[dim]Installed: {package_name}[/dim]")
            else:
                function_result = "SYSTEM ERROR: User denied permission."

        elif function_name == "update_tracker":
            markdown_content = args.get("markdown_content", "")
            existing = get_progress(working_dir)
            function_result = write_progress(working_dir, markdown_content)
            if existing and not existing.startswith("No PROGRESS.md"):
                function_result += f"\n\nPREVIOUS CONTENT (now overwritten):\n{existing}"
            console.print(f"[dim]Updated PROGRESS.md[/dim]")

        elif function_name == "ask_user":
            question = args.get("question", "")
            console.print("\n[bold]User Input Required:[/bold]")
            console.print(Markdown(question))
            user_feedback = console.input("\n[bold]Your response > [/bold]")
            if user_feedback.lower() in ['exit', 'quit']:
                return "Task aborted by user."
            function_result = f"USER RESPONSE: {user_feedback}"

        else:
            get_logger().log_error("unknown_tool", ToolNotFoundError(function_name))
            function_result = f"SYSTEM ERROR: Unknown tool '{function_name}' was called. This tool does not exist."

    except FileNotFoundError as e:
        get_logger().log_error(f"tool_{function_name}", e)
        function_result = f"Error: File not found — {e}"
    except PermissionError as e:
        get_logger().log_error(f"tool_{function_name}", e)
        function_result = f"Error: Permission denied — {e}"
    except Exception as e:
        get_logger().log_error(f"tool_{function_name}", e)
        function_result = f"Error: {type(e).__name__}: {e}"

    return function_result
