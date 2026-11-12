"""Keep secrets out of logs.

Telegram puts the bot token in the request path (https://api.telegram.org/bot<token>/sendMessage), and
httpx logs every request URL at INFO, so without this the token lands in the terminal and in the
server's journal. The filter rewrites any log line before it is printed: bot tokens become
`bot<redacted>`. It is installed on the root handlers by `tradeengine` at start-up.
"""

from __future__ import annotations

import logging
import re

_PATTERNS = [
    (re.compile(r"bot\d{5,}:[A-Za-z0-9_-]{20,}"), "bot<redacted>"),  # Telegram bot token
]


def redact(text: str) -> str:
    for pat, repl in _PATTERNS:
        text = pat.sub(repl, text)
    return text


class RedactFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        clean = redact(msg)
        if clean != msg:
            record.msg, record.args = clean, None
        if record.exc_info and record.exc_info[1] is not None:
            text = redact(logging.Formatter().formatException(record.exc_info))
            record.exc_info, record.exc_text = None, text
        return True


def install() -> None:
    root = logging.getLogger()
    for h in root.handlers:
        if not any(isinstance(f, RedactFilter) for f in h.filters):
            h.addFilter(RedactFilter())
