import json
from datetime import datetime

import psycopg
from psycopg.rows import dict_row

from src.storage.table_definitions import _CREATE_EVAL_RUNS_TABLE


class EvalStore:
    def __init__(self, conninfo: str) -> None:
        self._conninfo = conninfo
        self._create_tables()

    def _create_tables(self) -> None:
        with psycopg.connect(self._conninfo) as conn:
            conn.execute(_CREATE_EVAL_RUNS_TABLE)
            conn.commit()

    @staticmethod
    def _serialize_row(row: dict) -> dict:
        created_at: datetime | None = row.get("created_at")
        return {
            "id": int(row["id"]),
            "created_at": created_at.isoformat() if created_at else None,
            "config": row.get("config") or {},
            "retrieval_results": row.get("retrieval_results") or {},
            "answer_quality_results": row.get("answer_quality_results"),
        }

    def save_run(
        self,
        config: dict,
        retrieval_results: dict,
        answer_quality_results: dict | None = None,
    ) -> int:
        with psycopg.connect(self._conninfo) as conn:
            row = conn.execute(
                """
                INSERT INTO eval_runs (config, retrieval_results, answer_quality_results)
                VALUES (%s::jsonb, %s::jsonb, %s::jsonb)
                RETURNING id
                """,
                (
                    json.dumps(config),
                    json.dumps(retrieval_results),
                    json.dumps(answer_quality_results) if answer_quality_results is not None else None,
                ),
            ).fetchone()
            conn.commit()
        return int(row[0])

    def list_runs(self, limit: int = 20) -> list[dict]:
        with psycopg.connect(self._conninfo) as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """
                    SELECT id, created_at, config, retrieval_results, answer_quality_results
                    FROM eval_runs
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
                rows = cur.fetchall()
        return [self._serialize_row(row) for row in rows]

    def get_latest_run(self) -> dict | None:
        runs = self.list_runs(limit=1)
        return runs[0] if runs else None

    def get_run(self, run_id: int) -> dict | None:
        with psycopg.connect(self._conninfo) as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """
                    SELECT id, created_at, config, retrieval_results, answer_quality_results
                    FROM eval_runs
                    WHERE id = %s
                    """,
                    (run_id,),
                )
                row = cur.fetchone()
        return self._serialize_row(row) if row else None
