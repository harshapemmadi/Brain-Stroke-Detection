"""
Standalone DB setup script - run this when Django is available.
Creates SQLite tables for UserProfile and SegmentationResult.
"""
import sqlite3, os

DB_PATH = os.path.join(os.path.dirname(__file__), 'db.sqlite3')
conn = sqlite3.connect(DB_PATH)
c = conn.cursor()

c.execute('''CREATE TABLE IF NOT EXISTS TumorApp_userprofile (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    contact TEXT NOT NULL,
    address TEXT DEFAULT '',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
)''')

c.execute('''CREATE TABLE IF NOT EXISTS TumorApp_segmentationresult (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER REFERENCES TumorApp_userprofile(id) ON DELETE CASCADE,
    method TEXT NOT NULL,
    uploaded_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    accuracy REAL DEFAULT 0,
    probability REAL DEFAULT 0,
    has_tumor INTEGER DEFAULT 0,
    otsu_thresh REAL,
    improvement REAL,
    image_name TEXT DEFAULT ''
)''')

# Django session tables
c.execute('''CREATE TABLE IF NOT EXISTS django_session (
    session_key VARCHAR(40) PRIMARY KEY,
    session_data TEXT NOT NULL,
    expire_date DATETIME NOT NULL
)''')

conn.commit()
conn.close()
print(f"Database created at: {DB_PATH}")
