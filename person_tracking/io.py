"""Excel and video output."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from person_tracking.config import Config
from person_tracking.roi import ZoneDwellTracker


def write_excel(cfg: Config, gid_lifecycle: dict, zone: ZoneDwellTracker, source_fps: float):
    lifecycle_cols = [
        "global_id", "id_created_frame", "id_created_time_sec",
        "id_exit_frame", "id_exit_time_sec",
    ]
    rows = sorted(gid_lifecycle.values(), key=lambda r: r["global_id"]) if gid_lifecycle else []
    df_life = pd.DataFrame(rows, columns=lifecycle_cols) if rows else pd.DataFrame(columns=lifecycle_cols)

    zone_cols = [
        "global_id", "visit", "entry_frame", "entry_time_sec", "entry_time",
        "exit_frame", "exit_time_sec", "exit_time", "dwell_time_sec", "dwell_frames",
    ]
    zrows = zone.rows(source_fps)
    df_zone = pd.DataFrame(zrows, columns=zone_cols) if zrows else pd.DataFrame(columns=zone_cols)

    try:
        import openpyxl  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Excel export requires openpyxl. Run: pip install 'openpyxl>=3.1,<4'"
        ) from exc

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(cfg.excel_path, engine="openpyxl") as writer:
        df_life.to_excel(writer, index=False, sheet_name="id_lifecycle")
        df_zone.to_excel(writer, index=False, sheet_name="zone_dwell")
    print(f"[INFO] Excel: {cfg.excel_path} ({len(df_life)} IDs, {len(df_zone)} zone visits)")
