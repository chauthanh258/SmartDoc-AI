import logging
import os
import re
import sys


class _RedactionFilter(logging.Filter):
    """Mask common secret patterns before log records are emitted."""

    _patterns = [
        (re.compile(r"(OLLAMA_API_KEY\s*[=:]\s*)([^\s,;]+)", re.IGNORECASE), r"\1***"),
        (re.compile(r"(Authorization\s*[:=]\s*Bearer\s+)([^\s,;]+)", re.IGNORECASE), r"\1***"),
        (re.compile(r"(api[_-]?key\s*[=:]\s*)([^\s,;]+)", re.IGNORECASE), r"\1***"),
    ]

    def filter(self, record):
        message = record.getMessage()
        redacted = message
        for pattern, replacement in self._patterns:
            redacted = pattern.sub(replacement, redacted)

        if redacted != message:
            record.msg = redacted
            record.args = ()
        return True

def setup_logger(name="smartdoc", log_file="logs/smartdoc.log", level=logging.INFO):
    """Thiết lập logger với định dạng chi tiết."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers if setup_logger is called multiple times
    if not logger.handlers:
        logger.setLevel(level)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # File handler
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.addFilter(_RedactionFilter())
        logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.addFilter(_RedactionFilter())
        logger.addHandler(console_handler)
        
    return logger

# Single instance to be imported across modules
logger = setup_logger()
