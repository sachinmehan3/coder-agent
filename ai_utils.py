import litellm
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

litellm.drop_params = True

RETRY_EXCEPTIONS = (litellm.RateLimitError, litellm.ServiceUnavailableError, litellm.Timeout)

@retry(
    stop=stop_after_attempt(5), 
    wait=wait_exponential(multiplier=2, min=5, max=60), 
    retry=retry_if_exception_type(RETRY_EXCEPTIONS),
    before_sleep=lambda retry_state: print(f"  API rate limit or timeout. Retrying in {retry_state.next_action.sleep}s (attempt {retry_state.attempt_number}/5)...")
)
def safe_completion(model, messages, tools=None):
    """Unified LLM completion wrapper with automatic retry on transient errors."""
    kwargs = {
        "model": model,
        "messages": messages,
    }
    if tools:
        kwargs["tools"] = tools
    return litellm.completion(**kwargs)