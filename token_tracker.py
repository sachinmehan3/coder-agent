import litellm


class TokenTracker:
    """Tracks token usage and estimated cost across an entire agent session."""

    def __init__(self):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost = 0.0
        self.call_count = 0

    def record(self, response):
        """Extract usage from a LiteLLM response and accumulate totals."""
        usage = getattr(response, "usage", None)
        if not usage:
            return

        prompt = getattr(usage, "prompt_tokens", 0) or 0
        completion = getattr(usage, "completion_tokens", 0) or 0

        self.total_prompt_tokens += prompt
        self.total_completion_tokens += completion
        self.call_count += 1

        # Use litellm's built-in cost calculation
        try:
            call_cost = litellm.completion_cost(completion_response=response)
            self.total_cost += call_cost
        except Exception:
            pass  # Model not in pricing DB — skip cost

    @property
    def total_tokens(self):
        return self.total_prompt_tokens + self.total_completion_tokens

    def format_summary(self):
        """Returns a compact summary string for display at session end."""
        cost_str = f"${self.total_cost:.4f}" if self.total_cost > 0 else "unknown"
        return (
            f"Calls: {self.call_count}  |  "
            f"Tokens: {self.total_prompt_tokens:,} in / {self.total_completion_tokens:,} out  |  "
            f"Cost: {cost_str}"
        )


def get_max_context_tokens(model: str, fallback: int = 30000) -> int:
    """Returns ~80% of the model's context window, or a fallback if unknown."""
    try:
        info = litellm.get_model_info(model)
        max_input = info.get("max_input_tokens") or info.get("max_tokens") or fallback
        return int(max_input * 0.75)
    except Exception:
        return fallback
