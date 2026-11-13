
import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# SQLite database file
DATABASE_PATH = PROJECT_ROOT / "checkpoints.sqlite"

# Create SQLite connection
connection = sqlite3.connect(
    str(DATABASE_PATH),
    check_same_thread=False
)

# Initialize persistent LangGraph memory
memory = SqliteSaver(connection)

# Create checkpoint tables if they do not exist
memory.setup()
