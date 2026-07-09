from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path

from .response_record import ResponseRecord


class ResponseStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize_schema(self) -> None:
        with closing(self._connect()) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS response_records (
                    content_hash TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS numerical_results (
                    job_signature TEXT PRIMARY KEY,
                    content_hash TEXT NOT NULL,
                    FOREIGN KEY (content_hash) REFERENCES response_records (content_hash)
                )
                """
            )
            connection.commit()

    def put_record(self, record: ResponseRecord) -> None:
        with closing(self._connect()) as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO response_records (content_hash, payload_json)
                VALUES (?, ?)
                """,
                (record.content_hash, json.dumps(record.to_payload(), sort_keys=True)),
            )
            connection.commit()

    def get_record(self, content_hash: str) -> ResponseRecord | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT payload_json FROM response_records WHERE content_hash = ?",
                (content_hash,),
            ).fetchone()

        if row is None:
            return None

        return ResponseRecord.from_payload(json.loads(row["payload_json"]))

    def put_numerical_result(self, job_signature: str, record: ResponseRecord) -> None:
        self.put_record(record)
        with closing(self._connect()) as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO numerical_results (job_signature, content_hash)
                VALUES (?, ?)
                """,
                (str(job_signature), record.content_hash),
            )
            connection.commit()

    def get_numerical_result(self, job_signature: str) -> ResponseRecord | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT content_hash FROM numerical_results WHERE job_signature = ?",
                (str(job_signature),),
            ).fetchone()

        if row is None:
            return None

        return self.get_record(row["content_hash"])

    def list_numerical_signatures(self) -> list[str]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT job_signature FROM numerical_results ORDER BY job_signature"
            ).fetchall()
        return [row["job_signature"] for row in rows]

    def delete_numerical_result(self, job_signature: str) -> bool:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT content_hash FROM numerical_results WHERE job_signature = ?",
                (str(job_signature),),
            ).fetchone()
            if row is None:
                return False
            content_hash = row["content_hash"]
            connection.execute(
                "DELETE FROM numerical_results WHERE job_signature = ?",
                (str(job_signature),),
            )
            still_referenced = connection.execute(
                "SELECT 1 FROM numerical_results WHERE content_hash = ? LIMIT 1",
                (content_hash,),
            ).fetchone()
            if still_referenced is None:
                connection.execute(
                    "DELETE FROM response_records WHERE content_hash = ?",
                    (content_hash,),
                )
            connection.commit()
        return True