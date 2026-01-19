"""
Two-phase cone processing pipeline

PHASE 1:
    - Each cone is processed once
    - No inline retries
    - Failures are classified and persisted

PHASE 2:
    - Only transient failures are retried
    - Exponential backoff with cap
    - Permanent failures are recorded
"""

import time
import traceback
import logging
import pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path
from requests.exceptions import Timeout

from adaptive_dem_segment import dem_segment, NullError, DownloadError, DiskSpaceError
from measure import cone_metrics


# --- Configuration ---
POLYGON_FOLDER = Path(r"D:\Polygons")
DEM_FOLDER = Path(r"D:\DEMs")
VENT_COORD = Path(r"D:\vent_coords.xls")
CSV_OUT = Path(r"D:\Metrics.csv")

RUN_LOG = Path(r"D:\cone_run.log")
FAILURE_LOG = Path(r"D:\cone_failures.csv")

BASE_RETRY_DELAY = 60        # seconds
MAX_RETRY_DELAY = 300
MAX_TOTAL_ATTEMPTS = 5

TRANSIENT_ERRORS = (DownloadError, Timeout, TimeoutError)
FATAL_ERRORS = (NullError, DiskSpaceError)


# Logging setup
logging.basicConfig(
    filename=RUN_LOG,
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("cone_pipeline")


# --- Helper Functions ---
def compute_backoff(attempts: int) -> int:
    """Exponential backoff with cap."""
    return min(BASE_RETRY_DELAY * (2 ** (attempts - 1)), MAX_RETRY_DELAY)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_cone_record(id_, lat, lon) -> dict:
    return {
        "id": id_,
        "lat": lat,
        "lon": lon,
        "status": "PENDING",
        "attempts": 0,
        "error_type": None,
        "error_msg": None,
        "traceback": None,
        "last_attempt": None,
        "next_retry_after": None
    }


# --- Worker function ---
def process_cone_once(cone: dict) -> dict:
    """
    Process a cone exactly once.
    Returns updated cone record.
    """
    print(f"Processing cone {cone['id']} (attempt {cone['attempts'] + 1})")
    cone = cone.copy()
    cone["attempts"] += 1
    cone["last_attempt"] = utc_now()

    try:
        dem = dem_segment(
            cone["lat"], cone["lon"], cone["id"],
            POLYGON_FOLDER,
            DEM_FOLDER,
            diag=False
        )

        cone_dem, cone_poly, crater_poly, WARNING, warning_reasons = dem

        cone_metrics(
            lat=cone["lat"],
            lon=cone["lon"],
            num=cone["id"],
            cone_dem=cone_dem,
            cone_boundary=cone_poly,
            crater_boundary=crater_poly,
            WARNING=WARNING,
            warning_reasons=warning_reasons,
            output_csv=CSV_OUT,
            diag=False
        )

        cone["status"] = "SUCCESS"
        return cone

    except FATAL_ERRORS as e:
        cone["status"] = "FAILED_FATAL"
        cone["error_type"] = type(e).__name__
        cone["error_msg"] = str(e)
        cone["traceback"] = traceback.format_exc()
        return cone

    except TRANSIENT_ERRORS as e:
        cone["status"] = "RETRY_LATER"
        cone["error_type"] = type(e).__name__
        cone["error_msg"] = str(e)
        cone["traceback"] = traceback.format_exc()

    except Exception as e:
        cone["status"] = "RETRY_LATER"
        cone["error_type"] = type(e).__name__
        cone["error_msg"] = str(e)
        cone["traceback"] = traceback.format_exc()

    delay = compute_backoff(cone["attempts"])
    cone["next_retry_after"] = (
        datetime.now(timezone.utc) + timedelta(seconds=delay)
    ).isoformat()

    return cone


# --- Phase 1 ---
def phase_one(cones):
    logger.info("PHASE 1 STARTED")
    failures = []

    for cone in cones:
        res = process_cone_once(cone)

        if res["status"] == "SUCCESS":
            logger.info(f"Cone {res['id']} SUCCESS")
        else:
            failures.append(res)
            logger.warning(f"Cone {res['id']} -> {res['status']}")

    logger.info("PHASE 1 COMPLETE")
    return failures


# --- Phase 2 ---
def phase_two(failures):
    logger.info("PHASE 2 STARTED")
    active = failures

    while active:
        next_round = []
        now = datetime.now(timezone.utc)

        for cone in active:
            if cone["attempts"] >= MAX_TOTAL_ATTEMPTS:
                cone["status"] = "FAILED_FATAL"
                next_round.append(cone)
                continue

            if now < datetime.fromisoformat(cone["next_retry_after"]):
                next_round.append(cone)
                continue

            logger.info(
                f"Retrying cone {cone['id']} "
                f"(attempt {cone['attempts'] + 1})"
            )

            res = process_cone_once(cone)

            if res["status"] != "SUCCESS":
                next_round.append(res)
            else:
                logger.info(f"Cone {res['id']} RECOVERED")

        active = next_round

    logger.info("PHASE 2 COMPLETE")
    return active


# --- Main Execution ---
if __name__ == "__main__":
    start = time.perf_counter()
    print("Starting two-phase cone processing pipeline...")

    df = pd.read_excel(VENT_COORD)
    cones = [
        make_cone_record(r.ID, r.Latitude, r.Longitude)
        for r in df.itertuples()
    ]

    failures = phase_one(cones)

    if failures:
        pd.DataFrame(failures).to_csv(FAILURE_LOG, index=False)
        failures = phase_two(failures)
        pd.DataFrame(failures).to_csv(FAILURE_LOG, index=False)

    runtime = time.perf_counter() - start
    logger.info(f"PIPELINE COMPLETE | Runtime {runtime:.2f}s")
    print(f"PIPELINE COMPLETE ({runtime:.2f}s)")
