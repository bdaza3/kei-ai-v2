#Tools file for Kei to use.
from typing import List, Dict, Any
from monitor import ActivityMonitor, ActivitySnapshot, ActivityEntry


#List of tools that Kei can potentially use to interact with. 
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_active_window",
            "description": "Get the current active app and window title.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]

def get_active_window():
    monitor = ActivityMonitor()
    snapshot = monitor.sample()
    return {
        "active_app": snapshot.active_app,
        "active_window_title": snapshot.active_window_title,
        "active_process_name": snapshot.active_process_name,
        "current_category": snapshot.current_category,
        "idle_seconds": round(snapshot.idle_seconds, 1),
    }


