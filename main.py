import os
import argparse
from dotenv import load_dotenv
from rich.console import Console

from agent import get_initial_messages, run_agent_loop
from token_tracker import TokenTracker, get_max_context_tokens

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
    args = parser.parse_args()
    
    working_dir = args.dir
    model = args.model
    
    if not os.path.exists(working_dir):
        os.makedirs(working_dir)

    env_path = os.path.join(working_dir, ".env")
    if os.path.exists(env_path):
        load_dotenv(env_path, override=True)
    else:
        load_dotenv(os.path.join(os.getcwd(), ".env"), override=True)
    
    required_key = resolve_api_key_env(model)
    if required_key and not os.environ.get(required_key):
        print(f"ERROR: {required_key} not found in .env file (required for model '{model}').")
        return

    console = Console()
    
    console.print(f"[bold green] Workspace: {working_dir}[/bold green]")
    console.print(f"[bold green] Model: {model}[/bold green]")

    messages = get_initial_messages()
    tracker = TokenTracker()

    # Show detected context window
    detected_limit = get_max_context_tokens(model)
    console.print(f"[dim]Context limit: ~{detected_limit:,} tokens (75% of model max)[/dim]")
    
    console.print("[yellow]Starting...[/yellow]")
    console.print("[dim]Commands: /clear, /usage, exit[/dim]")

    while True:
        try:
            user_input = console.input("\n[bold blue]You > [/bold blue]")
            cmd = user_input.strip().lower()
            
            if cmd in ["exit", "quit"]:
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

            messages = run_agent_loop(model, console, working_dir, user_input, messages, tracker=tracker)
            console.print(f"[dim]{tracker.format_summary()}[/dim]")

        except Exception as e:
            from rich.markup import escape
            console.print(f"[bold red]System Error:[/bold red] {escape(str(e))}")
            break

if __name__ == "__main__":
    main()