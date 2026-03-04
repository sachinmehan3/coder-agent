"""Custom exception hierarchy for the coder-agent.

Allows the system to distinguish between retryable, fatal, and recoverable errors
instead of catching bare `Exception` everywhere.
"""


class AgentError(Exception):
    """Base exception for all agent-related errors."""
    pass


class ToolExecutionError(AgentError):
    """A tool failed during execution (file I/O error, subprocess crash, etc.)."""
    def __init__(self, tool_name: str, message: str):
        self.tool_name = tool_name
        super().__init__(f"Tool '{tool_name}' failed: {message}")


class ToolNotFoundError(AgentError):
    """The LLM called a tool that doesn't exist."""
    def __init__(self, tool_name: str):
        self.tool_name = tool_name
        super().__init__(f"Unknown tool '{tool_name}' was called. This tool does not exist.")


class SecurityViolationError(AgentError):
    """An operation was blocked for security reasons (path traversal, blocked file, etc.)."""
    def __init__(self, message: str):
        super().__init__(f"Security violation: {message}")


class CostLimitExceeded(AgentError):
    """The session exceeded its token or cost budget."""
    def __init__(self, limit_type: str, current, limit):
        self.limit_type = limit_type
        super().__init__(f"{limit_type} limit exceeded: {current} / {limit}")


class CircuitBreakerTripped(AgentError):
    """Too many consecutive failures of the same type — breaking the loop."""
    def __init__(self, tool_name: str, failure_count: int):
        self.tool_name = tool_name
        self.failure_count = failure_count
        super().__init__(
            f"Circuit breaker tripped: '{tool_name}' failed {failure_count} times consecutively."
        )
