import os
import json
import uuid
import logging
from datetime import datetime, timezone


class AgentLogger:
    """Structured JSON logger for the agent session.
    
    Writes one JSON object per line to logs/<session_id>.jsonl.
    Also logs WARNING+ to the console via the standard logging module.
    """

    def __init__(self, log_dir="logs", session_id=None):
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self.turn_id = 0
        self.log_dir = log_dir

        os.makedirs(log_dir, exist_ok=True)
        self._log_file = os.path.join(log_dir, f"{self.session_id}.jsonl")

        # Console logger for warnings/errors only
        self._console_logger = logging.getLogger(f"agent.{self.session_id}")
        if not self._console_logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(logging.WARNING)
            handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
            self._console_logger.addHandler(handler)
            self._console_logger.setLevel(logging.DEBUG)

    def next_turn(self):
        """Increment the turn counter. Call once per user input."""
        self.turn_id += 1

    def _write(self, event: str, level: str = "INFO", **data):
        """Write a structured JSON log entry to the log file."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "level": level,
            "event": event,
            **data
        }
        try:
            with open(self._log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception:
            pass  # Never let logging crash the agent

        # Mirror warnings and errors to console
        if level in ("WARNING", "ERROR"):
            getattr(self._console_logger, level.lower(), self._console_logger.warning)(
                f"{event}: {data}" if data else event
            )

    # --- High-level logging methods ---

    def log_llm_call(self, model, prompt_tokens=0, completion_tokens=0,
                     cost=0.0, latency_ms=0, source="agent"):
        """Log an LLM API call with usage details."""
        self._write(
            "llm_call",
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost=round(cost, 6),
            latency_ms=round(latency_ms, 1),
            source=source
        )

    def log_tool_call(self, tool_name, args_summary="", success=True,
                      result_preview="", source="agent"):
        """Log a tool invocation."""
        self._write(
            "tool_call",
            tool_name=tool_name,
            args_summary=str(args_summary)[:300],
            success=success,
            result_preview=str(result_preview)[:200],
            source=source
        )

    def log_error(self, context, error, source="agent"):
        """Log an error with context."""
        self._write(
            "error",
            level="ERROR",
            context=context,
            error_type=type(error).__name__,
            error_message=str(error)[:500],
            source=source
        )

    def log_session_start(self, model, working_dir):
        """Log the start of a session."""
        self._write(
            "session_start",
            model=model,
            working_dir=working_dir
        )

    def log_session_end(self, total_tokens, total_cost, call_count):
        """Log the end of a session."""
        self._write(
            "session_end",
            total_tokens=total_tokens,
            total_cost=round(total_cost, 6),
            call_count=call_count
        )

    def log_user_input(self, user_input):
        """Log user input (truncated for privacy)."""
        self._write(
            "user_input",
            content=user_input[:500]
        )

    def log_memory_trim(self, before_tokens, after_tokens):
        """Log a memory trim event."""
        self._write(
            "memory_trim",
            level="WARNING",
            before_tokens=before_tokens,
            after_tokens=after_tokens,
            tokens_freed=before_tokens - after_tokens
        )

    def save_messages(self, messages):
        """Auto-save conversation messages to disk for crash recovery.
        
        Writes to logs/<session_id>_messages.json after each turn.
        """
        save_path = os.path.join(self.log_dir, f"{self.session_id}_messages.json")
        try:
            # Filter to only serializable dict messages (skip any objects)
            serializable = []
            for msg in messages:
                if isinstance(msg, dict):
                    serializable.append(msg)
                else:
                    serializable.append({
                        "role": getattr(msg, "role", "unknown"),
                        "content": str(getattr(msg, "content", ""))
                    })
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(serializable, f, indent=2, default=str)
        except Exception:
            pass  # Never let auto-save crash the agent


# Module-level singleton — initialized by main.py, imported everywhere else
_logger = None


def init_logger(log_dir="logs", session_id=None):
    """Initialize the global logger. Call once from main.py."""
    global _logger
    _logger = AgentLogger(log_dir=log_dir, session_id=session_id)
    return _logger


def get_logger():
    """Get the global logger instance. Returns a no-op logger if not initialized."""
    global _logger
    if _logger is None:
        # Return a stub that silently does nothing — prevents crashes if
        # logger isn't initialized (e.g. during testing)
        _logger = AgentLogger(log_dir="logs")
    return _logger
