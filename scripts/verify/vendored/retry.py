"""
Shared retry helper with exponential backoff for all API calls.
Vendored verbatim from the earlier pipeline's retry.py (glyphs -> ASCII).
"""

import time
import logging

logger = logging.getLogger(__name__)


def with_retry(fn, max_attempts: int = 3, base_wait: float = 2.0):
    """Call fn() with exponential backoff retry; raise the last error."""
    last_exception = None

    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as e:
            last_exception = e
            if attempt < max_attempts - 1:
                wait = base_wait * (2 ** attempt)
                logger.warning(
                    f"API call failed (attempt {attempt + 1}/{max_attempts}): {e}. "
                    f"Retrying in {wait:.0f}s..."
                )
                print(f"  [retry] API call failed: {e}. Retrying in {wait:.0f}s...")
                time.sleep(wait)
            else:
                logger.error(f"API call failed after {max_attempts} attempts: {e}")
                print(f"  [fail] API call failed after {max_attempts} attempts: {e}")

    raise last_exception
