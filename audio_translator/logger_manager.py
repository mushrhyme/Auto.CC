# ---------- logger_manager.py ----------
import logging
from datetime import datetime
from constants import LOG_DIR

class LoggerManager:
    @staticmethod
    def setup_logger(name="TranslationLogger"):
        LOG_DIR.mkdir(exist_ok=True)
        log_filename = LOG_DIR / f"translation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        logger = logging.getLogger(name)
        if logger.handlers:
            for h in logger.handlers:
                logger.removeHandler(h)

        logger.setLevel(logging.INFO)
        fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        fh = logging.FileHandler(log_filename, encoding='utf-8')
        fh.setFormatter(fmt)
        logger.addHandler(fh)

        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)

        return logger