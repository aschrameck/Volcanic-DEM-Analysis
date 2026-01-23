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
from multiprocessing import Manager

from adaptive_dem_segment import dem_segment, NullError, DownloadError, DiskSpaceError
from measure import cone_metrics, CRS_Error


# --- Configuration ---
POLYGON_FOLDER = Path(r"D:\NASA_Research_Project\Tests\Metrics Test\Polygons")
DEM_FOLDER = Path(r"D:\NASA_Research_Project\Tests\Metrics Test\DEMs")
VENT_COORD = Path(r"D:\NASA_Research_Project\Tests\Metrics Test\test_vent_coords.xls")
CSV_OUT = Path(r"D:\NASA_Research_Project\Tests\Metrics Test\test_metrics.csv")

RUN_LOG = Path(r"D:\NASA_Research_Project\Tests\Metrics Test\cone_run.log")
FAILURE_LOG = Path(r"D:\NASA_Research_Project\Tests\Metrics Test\cone_failures.csv")

BASE_RETRY_DELAY = 30       # seconds
MAX_RETRY_DELAY = 300       # seconds
MAX_TOTAL_ATTEMPTS = 3      # maximum attempts per cone

TRANSIENT_ERRORS = (DownloadError, Timeout, TimeoutError)
FATAL_ERRORS = (NullError, DiskSpaceError, CRS_Error)


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
def process_cone_once(cone: dict, lock) -> dict:
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
            diag=False,
            lock=lock
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
def phase_one(cones, lock):
    logger.info("PHASE 1 STARTED")
    print("PHASE 1 STARTED")
    failures = []

    for cone in cones:
        res = process_cone_once(cone, lock)

        if res["status"] == "SUCCESS":
            logger.info(f"Cone {res['id']} SUCCESS")
            print(f"Cone {res['id']} SUCCESS")
        else:
            failures.append(res)
            logger.warning(f"Cone {res['id']} -> {res['status']} (reason: {res['error_type']})")
            print(f"Cone {res['id']} -> {res['status']} (reason: {res['error_type']})")

    logger.info("PHASE 1 COMPLETE")
    print("PHASE 1 COMPLETE")
    return failures


# --- Phase 2 ---
def phase_two(failures, lock):
    logger.info("PHASE 2 STARTED")
    print("PHASE 2 STARTED")

    active = [c for c in failures if c["status"] == "RETRY_LATER"]
    final_failures = [c for c in failures if c["status"] == "FAILED_FATAL"]

    while active:
        next_round = []
        now = datetime.now(timezone.utc)

        for cone in active:
            if cone["attempts"] >= MAX_TOTAL_ATTEMPTS:
                cone["status"] = "FAILED_FATAL"
                print(f"Cone {cone['id']} -> FAILED_FATAL (max attempts reached)")
                final_failures.append(cone)
                continue
            if cone["next_retry_after"] is not None:
                if now < datetime.fromisoformat(cone["next_retry_after"]):
                    next_round.append(cone)
                    continue

            logger.info(
                f"Retrying cone {cone['id']} "
                f"(attempt {cone['attempts'] + 1})"
            )

            res = process_cone_once(cone, lock)

            if res["status"] == "SUCCESS":
                logger.info(f"Cone {res['id']} RECOVERED")
            elif res["status"] == "FAILED_FATAL":
                logger.error(f"Cone {res['id']} -> FAILED_FATAL (reason: {res['error_type']})")
                final_failures.append(res)
            else:
                next_round.append(res)

        active = next_round

    logger.info("PHASE 2 COMPLETE")
    print("PHASE 2 COMPLETE")
    return final_failures


# --- Main Execution ---
if __name__ == "__main__":
    manager = Manager()
    METRICS_LOCK = manager.Lock()

    start = time.perf_counter()
    print("Starting two-phase cone processing pipeline...")

    df = pd.read_excel(VENT_COORD)
    cones = [
        make_cone_record(r.ID, r.Latitude, r.Longitude)
        for r in df.itertuples()
    ]

    failures = phase_one(cones, METRICS_LOCK)

    if failures:
        pd.DataFrame(failures).to_csv(FAILURE_LOG, index=False)
        failures = phase_two(failures, METRICS_LOCK)
        pd.DataFrame(failures).to_csv(FAILURE_LOG, index=False)

    runtime = time.perf_counter() - start
    logger.info(f"PIPELINE COMPLETE | Runtime {runtime:.2f}s")
    print(f"PIPELINE COMPLETE ({runtime:.2f}s)")
