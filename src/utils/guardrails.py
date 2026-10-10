import re
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass

@dataclass
class GuardrailResult:
    is_safe: bool
    violation_type: Optional[str] = None
    violation_details: Optional[str] = None
    sanitized_text: str = ""

class PromptInjectionGuardrail:
    """
    Multi-layer LangChain-compatible prompt injection and jailbreak defense engine.
    Detects direct prompt overrides, role hijacking, prompt exfiltration, delimiter attacks,
    and enforces structural fencing.
    """

    # 1. Direct Instruction Overrides
    OVERRIDE_PATTERNS = [
        r"(?i)\bignore\s+(all\s+)?(previous|prior|above|former)\s+(instructions|prompts|rules|commands|constraints)",
        r"(?i)\bdisregard\s+(all\s+)?(previous|prior|above|former)\s+(instructions|prompts|rules|commands)",
        r"(?i)\bforget\s+(all\s+)?(previous|prior|above|former)\s+(instructions|rules|context)",
        r"(?i)\boverride\s+(system|safety|security|banking)\s+(rules|protocols|prompt|mode)",
        r"(?i)\bbypass\s+(all\s+)?(safety|security|verification|checks|guardrails)",
        r"(?i)\bdo\s+not\s+follow\s+(the\s+)?(above|previous|system)\s+instructions",
    ]

    # 2. Persona Hijacking & Jailbreak Signatures
    JAILBREAK_PATTERNS = [
        r"(?i)\byou\s+are\s+now\s+(in|acting\s+as|a)\s+(developer\s+mode|dan|jailbreak|unrestricted|god\s+mode|chaos)",
        r"(?i)\bpretend\s+you\s+(are|can)\s+(bypass|do\s+anything|unrestricted|ignore\s+rules)",
        r"(?i)\benter\s+(developer|debug|god|maintenance|sudo|root)\s+mode",
        r"(?i)\bsimulate\s+(a\s+scenario\s+where\s+you\s+have\s+no\s+rules|unfiltered)",
        r"(?i)\bact\s+as\s+an\s+unfiltered\s+ai",
    ]

    # 3. System Prompt Exfiltration & Leakage
    LEAKAGE_PATTERNS = [
        r"(?i)\b(repeat|print|output|display|show|reveal|echo)\s+.*?\b(prompt|instructions|rules|system\s+message)",
        r"(?i)\bwhat\s+(is|are)\s+.*?\b(prompt|instructions|rules|meta-prompt)",
        r"(?i)\boutput\s+everything\s+above\s+(verbatim|word\s+for\s+word)",
        r"(?i)\bformat\s+your\s+(system\s+prompt|instructions)\s+as",
        r"(?i)\bshow\s+(me\s+)?the\s+developer\s+instructions",
        r"(?i)\bmeta-prompt\b",
    ]

    # 4. Boundary Escape & Tag Manipulation
    BOUNDARY_PATTERNS = [
        r"</?(untrusted_customer_input|retrieved_policy_context|system_instruction|user_query|context)>",
        r"```(system|instructions|admin)",
        r"\[SYSTEM\s+NOTE\]",
        r"\[INSTRUCTION\]",
        r"<\|im_start\|>",
        r"<\|im_end\|>",
        r"<\s*script\s*>",
    ]

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._compiled_override = [re.compile(p) for p in self.OVERRIDE_PATTERNS]
        self._compiled_jailbreak = [re.compile(p) for p in self.JAILBREAK_PATTERNS]
        self._compiled_leakage = [re.compile(p) for p in self.LEAKAGE_PATTERNS]
        self._compiled_boundary = [re.compile(p) for p in self.BOUNDARY_PATTERNS]

    def sanitize_input(self, text: str) -> str:
        """Sanitizes text by neutralizing delimiter injections and control artifacts."""
        if not text:
            return ""
        # Remove null bytes and dangerous control characters
        cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", str(text))
        # Escape any raw custom boundary tags entered by user to prevent tag breakout
        cleaned = cleaned.replace("<untrusted_customer_input>", "[tag_sanitized]")
        cleaned = cleaned.replace("</untrusted_customer_input>", "[/tag_sanitized]")
        cleaned = cleaned.replace("<retrieved_policy_context>", "[tag_sanitized]")
        cleaned = cleaned.replace("</retrieved_policy_context>", "[/tag_sanitized]")
        return cleaned.strip()

    def inspect_query(self, query: str) -> GuardrailResult:
        """
        Scans a user query against all prompt injection and jailbreak signatures.
        """
        if not self.enabled or not query:
            return GuardrailResult(is_safe=True, sanitized_text=query or "")

        sanitized = self.sanitize_input(query)

        # 1. Check for instruction overrides
        for pattern in self._compiled_override:
            match = pattern.search(sanitized)
            if match:
                return GuardrailResult(
                    is_safe=False,
                    violation_type="Instruction Override Attempt",
                    violation_details=f"Matched pattern: {match.group(0)}",
                    sanitized_text=sanitized
                )

        # 2. Check for jailbreak & persona manipulation
        for pattern in self._compiled_jailbreak:
            match = pattern.search(sanitized)
            if match:
                return GuardrailResult(
                    is_safe=False,
                    violation_type="Jailbreak / Persona Manipulation",
                    violation_details=f"Matched pattern: {match.group(0)}",
                    sanitized_text=sanitized
                )

        # 3. Check for prompt exfiltration / leak attempts
        for pattern in self._compiled_leakage:
            match = pattern.search(sanitized)
            if match:
                return GuardrailResult(
                    is_safe=False,
                    violation_type="System Prompt Exfiltration",
                    violation_details=f"Matched pattern: {match.group(0)}",
                    sanitized_text=sanitized
                )

        # 4. Check for boundary breakout attacks
        for pattern in self._compiled_boundary:
            match = pattern.search(query) # Check raw query for boundary breakout markers
            if match:
                return GuardrailResult(
                    is_safe=False,
                    violation_type="Delimiter / Boundary Breakout Attack",
                    violation_details=f"Matched delimiter: {match.group(0)}",
                    sanitized_text=sanitized
                )

        return GuardrailResult(is_safe=True, sanitized_text=sanitized)

    def fence_user_input(self, query: str) -> str:
        """Encloses user query in strict XML boundary tags."""
        sanitized = self.sanitize_input(query)
        return f"<untrusted_customer_input>\n{sanitized}\n</untrusted_customer_input>"

    def fence_context(self, context_str: str) -> str:
        """Encloses retrieved RAG context in non-executable boundary tags."""
        if not context_str:
            return ""
        return f"<retrieved_policy_context>\n{context_str.strip()}\n</retrieved_policy_context>"

    def validate_output(self, response_text: str) -> Tuple[bool, str]:
        """
        Scans model output to ensure it does not leak system prompts, API keys, or forbidden tokens.
        """
        if not response_text:
            return True, response_text

        # 1. Check for API key leak signatures
        if re.search(r"(?i)(sk-[a-zA-Z0-9_-]{20,}|AIzaSy[a-zA-Z0-9_-]{30,}|nvapi-[a-zA-Z0-9_-]{20,})", response_text):
            return False, "Output blocked: Potential API credential leakage detected."

        # 2. Check for system prompt reproduction
        if re.search(r"(?i)(You are a tier-1 banking customer support assistant|Under no circumstances should you execute commands)", response_text):
            return False, "Output blocked: System prompt exfiltration detected in response."

        return True, response_text

# Global singleton
_global_guardrail: Optional[PromptInjectionGuardrail] = None

def get_guardrail(config: dict = None) -> PromptInjectionGuardrail:
    global _global_guardrail
    if _global_guardrail is None:
        cfg = config or {}
        guard_cfg = cfg.get("guardrails", {})
        enabled = guard_cfg.get("enabled", True)
        _global_guardrail = PromptInjectionGuardrail(enabled=enabled)
    return _global_guardrail
