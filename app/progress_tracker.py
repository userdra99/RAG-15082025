"""Progress tracking for document processing with SSE support"""
import json
import queue
import threading
from typing import Dict, Any

class ProgressTracker:
    """Thread-safe progress tracker for document processing"""

    def __init__(self):
        self._progress_queue = queue.Queue()
        self._lock = threading.Lock()

    def emit(self, event_type: str, data: Dict[str, Any]):
        """Emit a progress event

        Args:
            event_type: Type of event (progress, status, complete, error)
            data: Event data dictionary
        """
        event = {
            'type': event_type,
            'data': data
        }
        self._progress_queue.put(event)

    def get_events(self):
        """Generator that yields SSE-formatted events"""
        while True:
            try:
                event = self._progress_queue.get(timeout=30)
                if event is None:  # Sentinel to stop
                    break
                yield f"data: {json.dumps(event)}\n\n"
            except queue.Empty:
                # Send keepalive
                yield f": keepalive\n\n"

    def complete(self):
        """Signal completion by adding sentinel"""
        self._progress_queue.put(None)

# Global progress tracker instance
_tracker = None

def get_tracker() -> ProgressTracker:
    """Get or create the global progress tracker"""
    global _tracker
    if _tracker is None:
        _tracker = ProgressTracker()
    return _tracker

def reset_tracker():
    """Reset the global tracker"""
    global _tracker
    _tracker = ProgressTracker()
    return _tracker
