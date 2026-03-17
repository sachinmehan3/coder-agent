"""Pydantic models for all agent tool arguments.

Each model defines the expected parameters for one tool. The JSON Schema
for the LLM is auto-generated via `model_json_schema()`, keeping
agent_tools.py in perfect sync with validation logic — zero hand-written
schemas, zero drift.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Tool argument models
# ---------------------------------------------------------------------------

class GetFilesInfoArgs(BaseModel):
    directory: Optional[str] = Field(
        default=".",
        description="The directory to inspect. Defaults to '.' for the whole project.",
    )


class GetFileContentArgs(BaseModel):
    file_path: str = Field(
        description="The EXACT path to the file you want to read. Do not guess this path.",
    )
    start_line: Optional[int] = Field(
        default=None,
        description="Optional. The line number to start reading from (1-indexed). Useful for reading large files in chunks.",
    )
    end_line: Optional[int] = Field(
        default=None,
        description="Optional. The line number to stop reading at (inclusive).",
    )


class CreateDirectoryArgs(BaseModel):
    directory_path: str = Field(
        description="The path of the directory to create (e.g., 'scripts', 'tests/unit').",
    )


class WriteFileArgs(BaseModel):
    file_path: str = Field(description="The path and filename to write to.")
    content: str = Field(description="The complete, final code for the file.")


class EditFileArgs(BaseModel):
    file_path: str = Field(
        description="The exact relative path to the file to edit.",
    )
    search: str = Field(
        description=(
            "The EXACT text block to find in the file. Copy this directly from "
            "`get_file_content` output. Include enough lines to uniquely identify the location."
        ),
    )
    replace: str = Field(
        description="The new text to replace the search block with.",
    )


class DeleteFileArgs(BaseModel):
    file_path: str = Field(
        description="The exact relative path of the file to delete.",
    )


class RunCompilerArgs(BaseModel):
    file_path: str = Field(
        description="The exact relative path to the Python file to compile.",
    )


class RunPythonFileArgs(BaseModel):
    file_path: str = Field(
        description="The exact relative path to the Python file to run.",
    )
    args: Optional[list[str]] = Field(
        default=None,
        description="Optional command-line arguments to pass to the script.",
    )


class WebSearchArgs(BaseModel):
    query: str = Field(description="The search query.")


class InstallPackageArgs(BaseModel):
    package_name: str = Field(
        description=(
            "The PyPI package name to install (e.g., 'requests', 'beautifulsoup4', "
            "'fastapi'). You can provide multiple packages separated by spaces."
        ),
    )


class AskUserArgs(BaseModel):
    question: str = Field(
        description=(
            "The question you want to ask the user, including any options "
            "or context they need."
        ),
    )


class UpdateTrackerArgs(BaseModel):
    markdown_content: str = Field(
        description=(
            "The COMPLETE markdown content for PROGRESS.md. "
            "Structure it with a project title, status, and checklist sections. "
            "Example:\n"
            "# Project: Build REST API\n"
            "## Status: In Progress\n"
            "## Completed\n"
            "- [x] Set up project structure\n"
            "## In Progress\n"
            "- [/] Implementing auth endpoints\n"
            "## Pending\n"
            "- [ ] Write unit tests\n"
        ),
    )


class SpawnSubagentArgs(BaseModel):
    task_description: str = Field(
        description=(
            "A detailed prompt describing what the sub-agent should accomplish. "
            "Be specific — include file names, requirements, and expected outcomes."
        ),
    )


class FinishTaskArgs(BaseModel):
    summary: str = Field(
        description="A summary of the actions you took, files you modified, and test results.",
    )


# ---------------------------------------------------------------------------
# Tool registry — single source of truth
# ---------------------------------------------------------------------------

class ToolDef:
    """Couples a tool's name, description, Pydantic arg model, and (optional)
    handler into one object so everything stays in sync."""

    def __init__(self, name: str, description: str, args_model: type[BaseModel]):
        self.name = name
        self.description = description
        self.args_model = args_model

    def to_openai_schema(self) -> dict:
        """Generate the OpenAI function-calling tool dict automatically."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.args_model.model_json_schema(),
            },
        }

    def parse_args(self, raw: dict) -> BaseModel:
        """Validate and parse raw dict into the typed model."""
        return self.args_model.model_validate(raw)


# ---------------------------------------------------------------------------
# Registry instances
# ---------------------------------------------------------------------------

TOOL_REGISTRY: dict[str, ToolDef] = {}


def _register(name: str, description: str, args_model: type[BaseModel]) -> ToolDef:
    td = ToolDef(name, description, args_model)
    TOOL_REGISTRY[name] = td
    return td


_register(
    "get_files_info",
    "Returns a recursive, complete map of all files and directories in the project. "
    "Use this first to understand the layout of the codebase.",
    GetFilesInfoArgs,
)

_register(
    "get_file_content",
    "Reads and returns the text content of a specified file. "
    "IMPORTANT: You must provide the EXACT relative path.",
    GetFileContentArgs,
)

_register(
    "create_directory",
    "Creates a new directory or a nested directory structure. "
    "Use this before creating files in a new folder.",
    CreateDirectoryArgs,
)

_register(
    "write_file",
    "Creates a new file or overwrites an existing file. "
    "You MUST provide the ENTIRE, complete file content from top to bottom.",
    WriteFileArgs,
)

_register(
    "edit_file",
    "Edit an existing file by replacing a specific block of text with new text. "
    "PREFERRED over `write_file` for modifying existing files — saves tokens and reduces errors. "
    "First tries exact match, then falls back to fuzzy matching. "
    "IMPORTANT: Copy the search text EXACTLY from the file including all whitespace and indentation. "
    "The search string must uniquely identify the block to replace (include enough surrounding lines if needed).",
    EditFileArgs,
)

_register(
    "delete_file",
    "Deletes a specific existing file. Use this to clean up unnecessary files or when restructuring.",
    DeleteFileArgs,
)

_register(
    "run_compiler",
    "Compiles a Python file (using py_compile) to check for syntax errors WITHOUT executing the code. "
    "Always use this to check your work before using run_python_file.",
    RunCompilerArgs,
)

_register(
    "run_python_file",
    "Executes a Python script and returns the console output (STDOUT and STDERR). "
    "CAUTION: NEVER execute GUI applications or blocking servers. "
    "If the script contains a GUI (e.g. tkinter, PyQt), test it strictly by using the run_compiler tool instead of running it.",
    RunPythonFileArgs,
)

_register(
    "web_search",
    "Searches the web for up-to-date information, documentation, or tutorials.",
    WebSearchArgs,
)

_register(
    "install_package",
    "Installs third-party Python packages using 'uv add'. "
    "Use this immediately if you encounter a ModuleNotFoundError when running a test.",
    InstallPackageArgs,
)

_register(
    "ask_user",
    "Stop and ask the user a question. Use this if you need clarification "
    "on requirements, design decisions, or if you are repeatedly failing and need human help.",
    AskUserArgs,
)

_register(
    "update_tracker",
    "Create or update the project progress tracker (PROGRESS.md). "
    "Write the FULL markdown content for the file. Use this at the START of a project "
    "to record the goal and milestones, and after completing work to update the status. "
    "Use markdown checklists: `- [x]` for done, `- [/]` for in-progress, `- [ ]` for pending.",
    UpdateTrackerArgs,
)

_register(
    "spawn_subagent",
    "Spawn an isolated sub-agent to handle a complex, self-contained subtask. "
    "The sub-agent gets its own context and tools, executes autonomously, and returns a summary. "
    "Use this for large refactors, research tasks, or any work you want to delegate "
    "without cluttering your main context.",
    SpawnSubagentArgs,
)

_register(
    "finish_task",
    "Call this tool ONLY when you have fully completed the assigned task. "
    "This ends your execution and returns your summary to the main agent.",
    FinishTaskArgs,
)
