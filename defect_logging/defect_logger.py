"""
Thread-safe SQLite logger for defect events and inspection sessions.
"""
import os
import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import pandas as pd


class DefectLogger:
    def __init__(self, db_path: str = "logs/inspection.db", snapshot_dir: str = "logs/snapshots"):
        self.db_path = Path(db_path)
        self.snapshot_dir = Path(snapshot_dir)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _get_conn(self):
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._lock:
            conn = self._get_conn()
            try:
                cur = conn.cursor()
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id TEXT PRIMARY KEY,
                        start_time TEXT,
                        end_time TEXT,
                        fabric_id TEXT,
                        fabric_length_m REAL,
                        frames_processed INTEGER DEFAULT 0,
                        total_defects INTEGER DEFAULT 0,
                        notes TEXT
                    )
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT,
                        event_id INTEGER,
                        defect_class TEXT,
                        confidence REAL,
                        first_seen TEXT,
                        last_seen TEXT,
                        first_frame INTEGER,
                        last_frame INTEGER,
                        frames_seen INTEGER DEFAULT 1,
                        bbox_x1 INTEGER,
                        bbox_y1 INTEGER,
                        bbox_x2 INTEGER,
                        bbox_y2 INTEGER,
                        snapshot_path TEXT,
                        FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                    )
                """)
                cur.execute("CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id)")
                conn.commit()
            finally:
                conn.close()

    def start_session(self, fabric_id: str = "roll-1", fabric_length_m: Optional[float] = None, notes: str = "") -> str:
        session_id = f"sess_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        start_time = datetime.now().isoformat()
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute("""
                    INSERT INTO sessions (session_id, start_time, fabric_id, fabric_length_m, notes)
                    VALUES (?, ?, ?, ?, ?)
                """, (session_id, start_time, fabric_id, fabric_length_m, notes))
                conn.commit()
            finally:
                conn.close()
        return session_id

    def log_event(self, session_id: str, event: dict, frame=None) -> Optional[str]:
        """Save a confirmed defect event and optional snapshot."""
        now_str = datetime.now().isoformat()
        first_seen = event.get("first_seen", now_str)
        if isinstance(first_seen, (int, float)):
            first_seen = datetime.fromtimestamp(first_seen).isoformat()
            
        bbox = event.get("bbox") or [0, 0, 0, 0]
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        
        snapshot_rel_path = ""
        # Save snapshot image if provided
        if frame is not None and frame.size > 0:
            ev_id = event.get("event_id", int(time.time() * 1000) % 100000)
            fname = f"snap_{session_id}_{ev_id}_{event.get('class', 'defect')}.jpg"
            snap_file = self.snapshot_dir / fname
            
            # Draw bbox and label on the snapshot copy
            vis = frame.copy()
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            label = f"{event.get('class', 'defect')} {event.get('confidence', 0.0):.2f}"
            cv2.putText(vis, label, (int(x1), max(20, int(y1) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imwrite(str(snap_file), vis)
            snapshot_rel_path = str(snap_file)

        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute("""
                    INSERT INTO events (
                        session_id, event_id, defect_class, confidence, first_seen, last_seen,
                        first_frame, last_frame, frames_seen, bbox_x1, bbox_y1, bbox_x2, bbox_y2, snapshot_path
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    session_id,
                    event.get("event_id", 0),
                    event.get("class", "defect"),
                    float(event.get("confidence", 0.0)),
                    str(first_seen),
                    str(event.get("last_seen", first_seen)),
                    int(event.get("first_frame", 0)),
                    int(event.get("last_frame", event.get("first_frame", 0))),
                    int(event.get("frames_seen", 1)),
                    int(x1), int(y1), int(x2), int(y2),
                    snapshot_rel_path
                ))
                # Update session defect count
                conn.execute("""
                    UPDATE sessions SET total_defects = total_defects + 1 WHERE session_id = ?
                """, (session_id,))
                conn.commit()
            finally:
                conn.close()
        return snapshot_rel_path

    def close_event(self, session_id: str, event: dict):
        last_seen = event.get("last_seen", datetime.now().isoformat())
        if isinstance(last_seen, (int, float)):
            last_seen = datetime.fromtimestamp(last_seen).isoformat()
            
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute("""
                    UPDATE events SET 
                        last_seen = ?,
                        last_frame = ?,
                        frames_seen = ?
                    WHERE session_id = ? AND event_id = ?
                """, (
                    str(last_seen),
                    int(event.get("last_frame", 0)),
                    int(event.get("frames_seen", 1)),
                    session_id,
                    event.get("event_id", 0)
                ))
                conn.commit()
            finally:
                conn.close()

    def end_session(self, session_id: str, frames_processed: int = 0):
        end_time = datetime.now().isoformat()
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute("""
                    UPDATE sessions SET 
                        end_time = ?,
                        frames_processed = ?
                    WHERE session_id = ?
                """, (end_time, int(frames_processed), session_id))
                conn.commit()
            finally:
                conn.close()

    def events_df(self, session_id: Optional[str] = None) -> pd.DataFrame:
        with self._lock:
            conn = self._get_conn()
            try:
                if session_id:
                    df = pd.read_sql_query(
                        "SELECT * FROM events WHERE session_id = ? ORDER BY id DESC",
                        conn, params=(session_id,)
                    )
                else:
                    df = pd.read_sql_query("SELECT * FROM events ORDER BY id DESC", conn)
                return df
            finally:
                conn.close()

    def sessions_df(self) -> pd.DataFrame:
        with self._lock:
            conn = self._get_conn()
            try:
                return pd.read_sql_query("SELECT * FROM sessions ORDER BY start_time DESC", conn)
            finally:
                conn.close()

    def export_csv(self, session_id: Optional[str] = None, out_path: Optional[str] = None) -> str:
        df = self.events_df(session_id)
        if not out_path:
            tag = session_id or "all_sessions"
            out_path = f"logs/defects_{tag}.csv"
        df.to_csv(out_path, index=False)
        return out_path

    def close(self):
        pass
