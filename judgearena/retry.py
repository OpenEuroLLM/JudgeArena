"""Retry classification shared by model backends."""

import json


def _is_retryable_error(e: Exception) -> bool:
    """Return True if the exception is a transient server error that should be retried.

    Handles three formats:
    - String representation contains the HTTP code (most providers)
    - ValueError raised by langchain-openai with a dict arg: {'message': ..., 'code': 429}
    - json.JSONDecodeError when a gateway returns a non-JSON (e.g. HTML) body
    """
    # langchain-openai raises ValueError(response_dict.get("error")) where the
    # error value is a dict like {'message': '...', 'code': 408}
    _RETRYABLE_CODES = {408, 429, 502, 503, 504}
    if isinstance(e, ValueError) and e.args:
        arg = e.args[0]
        if isinstance(arg, dict) and arg.get("code") in _RETRYABLE_CODES:
            return True

    # Gateways (e.g. OpenRouter) intermittently return non-JSON error bodies that
    # surface as JSONDecodeError while parsing the response. These are transient
    # and safe to retry. JSONDecodeError subclasses ValueError but carries a
    # string arg, so it bypasses the dict-code check above.
    if isinstance(e, json.JSONDecodeError):
        return True

    error_str = str(e)
    return (
        any(str(code) in error_str for code in _RETRYABLE_CODES)
        or "rate" in error_str.lower()
        or "Expecting value" in error_str
        or "JSONDecodeError" in error_str
    )
