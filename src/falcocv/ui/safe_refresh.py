import functools
import sys
import traceback

def safe_refresh(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except RuntimeError as exc:
            # A config dialog may still have a 100 ms preview timer queued when
            # it is closed.  Touching its deleted Qt widgets is harmless and
            # must not pollute the log.
            if "has been deleted" in str(exc):
                return
            print(f"[ERROR] Unhandled in {func.__qualname__}:", file=sys.stderr)
            traceback.print_exc()
        except Exception:
            print(f"[ERROR] Unhandled in {func.__qualname__}:", file=sys.stderr)
            traceback.print_exc()
    return wrapper
