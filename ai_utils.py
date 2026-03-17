import litellm
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

litellm.drop_params = True

# Only retry on transient errors — AuthenticationError is fatal, don't waste time retrying
RETRY_EXCEPTIONS = (litellm.RateLimitError, litellm.ServiceUnavailableError, litellm.Timeout)

def _log_retry(retry_state):
    """Log each retry attempt for observability."""
    print(f"  API rate limit or timeout. Retrying in {retry_state.next_action.sleep}s (attempt {retry_state.attempt_number}/5)...")

@retry(
    stop=stop_after_attempt(5), 
    wait=wait_exponential(multiplier=2, min=5, max=60), 
    retry=retry_if_exception_type(RETRY_EXCEPTIONS),
    before_sleep=_log_retry
)
def safe_completion(model, messages, tools=None):
    """Unified LLM completion wrapper with automatic retry on transient errors.
    
    Retries on: RateLimitError, ServiceUnavailableError, Timeout
    Fails fast on: AuthenticationError, BadRequestError, NotFoundError
    """
    kwargs = {
        "model": model,
        "messages": messages,
    }
    if tools:
        kwargs["tools"] = tools
    return litellm.completion(**kwargs)