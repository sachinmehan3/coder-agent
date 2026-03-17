"""Tool lists for the main agent and sub-agent.

All tool schemas are auto-generated from the Pydantic models in
tool_models.py — no hand-written JSON schemas to maintain.
"""

from tool_models import TOOL_REGISTRY

# Tools available to both agents (everything except spawn_subagent & finish_task)
_BASE_TOOL_NAMES = [
    "get_files_info",
    "get_file_content",
    "create_directory",
    "write_file",
    "edit_file",
    "delete_file",
    "run_compiler",
    "run_python_file",
    "web_search",
    "install_package",
    "ask_user",
    "update_tracker",
]

BASE_TOOLS = [TOOL_REGISTRY[name].to_openai_schema() for name in _BASE_TOOL_NAMES]

# Main agent gets base tools + spawn_subagent
AGENT_TOOLS = BASE_TOOLS + [TOOL_REGISTRY["spawn_subagent"].to_openai_schema()]

# Sub-agent gets base tools + finish_task (no spawn_subagent — prevents infinite nesting)
SUBAGENT_TOOLS = BASE_TOOLS + [TOOL_REGISTRY["finish_task"].to_openai_schema()]
