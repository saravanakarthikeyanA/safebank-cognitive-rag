import time
import threading
from typing import List, Dict, Tuple, Optional

def count_tokens(text: str) -> int:
    """
    Fast, reliable token estimation (~4 chars per token / 0.75 words per token).
    Works universally across Gemini, OpenAI, Groq, and NVIDIA without heavy binary dependencies.
    """
    if not text:
        return 0
    text_str = str(text).strip()
    if not text_str:
        return 0
    # Average across character-length heuristics and whitespace word count
    char_count = len(text_str)
    word_count = len(text_str.split())

    # Standard rule of thumb: ~4 characters per token in English, ~1.3 tokens per word
    est_from_chars = char_count / 4.0
    est_from_words = word_count * 1.3
    return max(1, int(round((est_from_chars + est_from_words) / 2.0)))

def truncate_text_to_tokens(text: str, max_tokens: int) -> str:
    """
    Truncates text to fit within a given token budget while preserving word boundaries.
    """
    if not text or max_tokens <= 0:
        return ""

    tokens = count_tokens(text)
    if tokens <= max_tokens:
        return text

    words = text.split()
    trimmed_words = []

    for word in words:
        candidate = " ".join(trimmed_words + [word]) + "..."
        if count_tokens(candidate) > max_tokens:
            break
        trimmed_words.append(word)

    if not trimmed_words:
        return text[:max(1, max_tokens * 3)] + "..."

    return " ".join(trimmed_words) + "..."

def budget_context_chunks(chunks: List[Dict], max_tokens: int = 450) -> Tuple[List[Dict], str, int]:
    """
    Selects and truncates retrieved document chunks to strictly fit within the token ceiling.

    Args:
        chunks: List of dicts with 'text', 'metadata', and optional 'score'.
        max_tokens: Maximum total tokens allowed for the combined context.

    Returns:
        Tuple of (selected_chunks, combined_context_str, total_tokens_used).
    """
    if not chunks or max_tokens <= 0:
        return [], "", 0

    selected_chunks = []
    current_tokens = 0
    context_parts = []

    for chunk in chunks:
        raw_text = chunk.get("text", "").strip()
        if not raw_text:
            continue

        chunk_token_count = count_tokens(raw_text)
        remaining_budget = max_tokens - current_tokens

        if remaining_budget <= 0:
            break

        if chunk_token_count <= remaining_budget:
            selected_chunks.append(chunk)
            context_parts.append(raw_text)
            current_tokens += chunk_token_count
        else:
            # Partially fit the last chunk if remaining budget allows meaningful content
            if remaining_budget >= 40:
                truncated_text = truncate_text_to_tokens(raw_text, remaining_budget)
                chunk_copy = dict(chunk)
                chunk_copy["text"] = truncated_text
                chunk_copy["truncated"] = True
                selected_chunks.append(chunk_copy)
                context_parts.append(truncated_text)
                current_tokens += count_tokens(truncated_text)
            break

    combined_str = "\n\n".join(context_parts)
    return selected_chunks, combined_str, current_tokens

def budget_chat_history(messages: List[Dict], max_tokens: int = 200) -> Tuple[List[Dict], str, int]:
    """
    Extracts the most recent conversation messages that fit within max_tokens budget.
    Iterates in reverse chronological order so the newest messages are prioritized.

    Args:
        messages: List of message dicts {'role': 'user'|'assistant', 'content': '...'}
        max_tokens: Maximum total tokens allowed for chat history.

    Returns:
        Tuple of (budgeted_messages, history_string, total_tokens_used)
    """
    if not messages or max_tokens <= 0:
        return [], "", 0

    selected_reversed = []
    current_tokens = 0

    # Iterate backwards from newest to oldest
    for msg in reversed(messages):
        role = msg.get("role", "user")
        content = msg.get("content", "").strip()
        if not content:
            continue

        line_repr = f"{role.capitalize()}: {content}\n"
        msg_tokens = count_tokens(line_repr)

        if current_tokens + msg_tokens <= max_tokens:
            selected_reversed.append(msg)
            current_tokens += msg_tokens
        else:
            # If we haven't included anything yet and the latest message is slightly long, truncate it
            if not selected_reversed and max_tokens >= 30:
                trimmed_content = truncate_text_to_tokens(content, max_tokens - 10)
                selected_reversed.append({"role": role, "content": trimmed_content})
                current_tokens = count_tokens(f"{role.capitalize()}: {trimmed_content}\n")
            break

    # Restore chronological order
    budgeted_messages = list(reversed(selected_reversed))
    history_str = "".join(f"{m['role'].capitalize()}: {m['content']}\n" for m in budgeted_messages)
    return budgeted_messages, history_str, current_tokens

class TokenBucketRateLimiter:
    """
    Thread-safe rate limiter implementing sliding window and min-interval throttling.
    Prevents HTTP 429 (Rate Limit Exceeded) errors on free/shared tiers.
    """
    def __init__(self, max_rpm: int = 15, min_interval_seconds: float = 1.0, enabled: bool = True):
        self.max_rpm = max_rpm
        self.min_interval_seconds = min_interval_seconds
        self.enabled = enabled
        self._timestamps: List[float] = []
        self._last_request_time: float = 0.0
        self._lock = threading.Lock()

    def acquire(self) -> float:
        """
        Blocks if necessary until a request token is available.
        Returns the duration waited in seconds.
        """
        if not self.enabled or self.max_rpm <= 0:
            return 0.0

        with self._lock:
            now = time.time()
            waited = 0.0

            # 1. Enforce minimum interval between consecutive requests
            elapsed_since_last = now - self._last_request_time
            if elapsed_since_last < self.min_interval_seconds:
                sleep_time = self.min_interval_seconds - elapsed_since_last
                time.sleep(sleep_time)
                waited += sleep_time
                now = time.time()

            # 2. Enforce sliding window RPM limit (last 60 seconds)
            cutoff = now - 60.0
            self._timestamps = [t for t in self._timestamps if t > cutoff]

            if len(self._timestamps) >= self.max_rpm:
                oldest = self._timestamps[0]
                sleep_time = max(0.0, (oldest + 60.0) - now + 0.1)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    waited += sleep_time
                    now = time.time()
                # Re-clean timestamps after sleeping
                cutoff = now - 60.0
                self._timestamps = [t for t in self._timestamps if t > cutoff]

            self._timestamps.append(now)
            self._last_request_time = now
            return waited

# Global singleton rate limiter for LLM calls
_global_rate_limiter: Optional[TokenBucketRateLimiter] = None

def get_rate_limiter(config: dict = None) -> TokenBucketRateLimiter:
    global _global_rate_limiter
    if _global_rate_limiter is None:
        cfg = config or {}
        tb_cfg = cfg.get("token_budget", {})
        rl_cfg = tb_cfg.get("rate_limiting", {})
        enabled = rl_cfg.get("enabled", True)
        max_rpm = int(rl_cfg.get("max_requests_per_minute", 15))
        min_interval = float(rl_cfg.get("min_request_interval_seconds", 1.0))
        _global_rate_limiter = TokenBucketRateLimiter(max_rpm=max_rpm, min_interval_seconds=min_interval, enabled=enabled)
    return _global_rate_limiter
