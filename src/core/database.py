import sqlite3
import json
from datetime import datetime

class DatabaseManager:
    def __init__(self, db_path="smartdoc_chat.db"):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        """Khởi tạo cấu trúc bảng nếu chưa có."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Bảng lưu phiên chat
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    title TEXT,
                    created_at TIMESTAMP
                )
            ''')
            # Bảng lưu chi tiết tin nhắn
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS chat_turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    question TEXT,
                    answer TEXT,
                    sources TEXT,  -- Lưu JSON string của list sources
                    timestamp TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id)
                )
            ''')
            conn.commit()

    def save_chat_turn(self, session_id, turn_data):
        """
        Lưu một lượt chat từ file chat_history vào DB.
        turn_data chính là 'entry' trả về từ hàm add_chat_turn.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO chat_turns (session_id, question, answer, sources, timestamp)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                session_id,
                turn_data['question'],
                turn_data['answer'],
                json.dumps(turn_data['sources']), # Convert list sang JSON string
                turn_data['timestamp']
            ))
            conn.commit()

    def get_session_history(self, session_id):
        """Lấy lịch sử của một session để nạp lại vào UI."""
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row # Để lấy dữ liệu dạng dict
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM chat_turns WHERE session_id = ? ORDER BY timestamp ASC', (session_id,))
            rows = cursor.fetchall()
            
            history = []
            for row in rows:
                history.append({
                    "question": row['question'],
                    "answer": row['answer'],
                    "sources": json.loads(row['sources']), # Convert ngược từ JSON string sang list
                    "timestamp": row['timestamp']
                })
            return history

    
    def create_session(self, session_id, title="Cuộc hội thoại mới"):
        """Tạo một session mới trong DB."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Chỉ chèn nếu chưa tồn tại
            cursor.execute('''
                INSERT OR IGNORE INTO sessions (session_id, title, created_at)
                VALUES (?, ?, ?)
            ''', (session_id, title, datetime.now().isoformat()))
            conn.commit()

    def get_all_sessions(self):
        """Lấy danh sách tất cả các cuộc hội thoại để hiện lên Sidebar."""
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM sessions ORDER BY created_at DESC')
            return cursor.fetchall()