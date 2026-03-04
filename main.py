import os
import argparse
from dotenv import load_dotenv
from rich.console import Console

from agent import get_initial_messages, run_agent_loop
from token_tracker import TokenTracker, get_max_context_tokens
from logger import init_logger

PROVIDER_KEY_MAP = {
    "mistral/": "MISTRAL_API_KEY",
    "openai/": "OPENAI_API_KEY",
    "anthropic/": "ANTHROPIC_API_KEY",
    "claude-": "ANTHROPIC_API_KEY",
    "gemini/": "GEMINI_API_KEY",
}

def resolve_api_key_env(model_name):
    """Returns the env var name required for the given model, or None if not needed."""
    for prefix, env_var in PROVIDER_KEY_MAP.items():
        if model_name.startswith(prefix):
            return env_var
    return None

def main():
    parser = argparse.ArgumentParser(description="CLI Coding Assistant")
    parser.add_argument("--dir", type=str, default="workspace", help="The directory the agent will work in.")
    parser.add_argument("--model", type=str, default="mistral/mistral-medium-latest", help="LiteLLM model identifier (e.g. gpt-4o, anthropic/claude-sonnet-4-20250514, mistral/mistral-medium-latest)")
    parser.add_argument("--log-dir", type=str, default="logs", help="Directory for session logs.")
    args = parser.parse_args()
    
    working_dir = args.dir
    model = args.model
    
    if not os.path.exists(working_dir):
        os.makedirs(working_dir)

    load_dotenv()
    
    required_key = resolve_api_key_env(model)
    if required_key and not os.environ.get(required_key):
        print(f"ERROR: {required_key} not found in .env file (required for model '{model}').")
        return

    console = Console()
    logger = init_logger(log_dir=args.log_dir)
    
    console.print(f"[bold green] Workspace: {working_dir}[/bold green]")
    console.print(f"[bold green] Model: {model}[/bold green]")

    messages = get_initial_messages()
    tracker = TokenTracker()

    # Show detected context window
    detected_limit = get_max_context_tokens(model)
    console.print(f"[dim]Context limit: ~{detected_limit:,} tokens (75% of model max)[/dim]")
    console.print(f"[dim]Session log: {logger._log_file}[/dim]")

    logger.log_session_start(model=model, working_dir=working_dir)

    console.print("[yellow]Starting...[/yellow]")
    console.print("[dim]Commands: /clear, /usage, exit[/dim]")

    while True:
        try:
            user_input = console.input("\n[bold blue]You > [/bold blue]")
            cmd = user_input.strip().lower()
            
            if cmd in ["exit", "quit"]:
                logger.log_session_end(
                    total_tokens=tracker.total_tokens,
                    total_cost=tracker.total_cost,
                    call_count=tracker.call_count
                )
                console.print(f"\n[bold]Session Summary:[/bold]")
                console.print(f"[dim]{tracker.format_summary()}[/dim]")
                break
                
            elif cmd == "/clear":
                os.system('cls' if os.name == 'nt' else 'clear')
                continue

            elif cmd == "/usage":
                console.print(f"\n[dim]{tracker.format_summary()}[/dim]")
                continue

            if not cmd:
                continue

            logger.next_turn()
            logger.log_user_input(user_input)
            messages = run_agent_loop(model, console, working_dir, user_input, messages, tracker=tracker)
            console.print(f"[dim]{tracker.format_summary()}[/dim]")

        except Exception as e:
            from rich.markup import escape
            logger.log_error("main_loop", e)
            console.print(f"[bold red]System Error:[/bold red] {escape(str(e))}")
            break

if __name__ == "__main__":
    main()