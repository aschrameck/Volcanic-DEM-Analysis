"""
Two-phase parallel cone processing pipeline with resume support
---------------------------------------------------------------

Features:
1. Automatic resume from previous failures CSV.
2. Phase 1: Each cone is processed once (new cones only).
3. Phase 2: Only transient failures are retried with exponential backoff.
4. Persistent failure log (CSV) updates after each phase.
5. Retry counters and backoff with cap.
6. Parallel processing using ProcessPoolExecutor.
7. Centralized logging for full run.
8. Progress bars using tqdm.
"""

import time
import traceback
import logging
import pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path
from requests.exceptions import Timeout
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import os

from adaptive_dem_segment import dem_segment, NullError, DownloadError, DiskSpaceError
from measure import cone_metrics

# --- Configuration ---
POLYGON_FOLDER = Path(r"D:\NASA_Research_Project\Tests\Metrics Test\Polygons")
DEM_FOLDER = Path(r"D:\NASA_Research_Project\Tests\Metrics Test\DEMs")
VENT_COORD = Path(r"D:\NASA_Research_Project\Tests\Metrics Test\test_vent_coords.xls")
CSV_OUT = Path(r"D:\NASA_Research_Project\Tests\Metrics Test\test_metrics.csv")

RUN_LOG = Path(r"D:\NASA_Research_Project\Tests\Metrics Test\cone_run.log")
FAILURE_LOG = Path(r"D:\NASA_Research_Project\Tests\Metrics Test\cone_failures.csv")

BASE_RETRY_DELAY = 60       # seconds
MAX_RETRY_DELAY = 300       # seconds
MAX_TOTAL_ATTEMPTS = 5      # maximum attempts per cone

MAX_WORKERS = max(1, os.cpu_count() - 3)  # leave 3 CPUs free

# Error classification
TRANSIENT_ERRORS = (DownloadError, Timeout, TimeoutError)
FATAL_ERRORS = (NullError, DiskSpaceError)

# --- Logging ---
logging.basicConfig(
    filename=RUN_LOG,
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("cone_pipeline")


# --- Helper Functions ---
def compute_backoff(attempts: int) -> int:
    """Compute exponential backoff (seconds) with cap."""
    return min(BASE_RETRY_DELAY * (2 ** (attempts - 1)), MAX_RETRY_DELAY)


def utc_now() -> str:
    """Return current UTC time as ISO string."""
    return datetime.now(timezone.utc).isoformat()


def make_cone_record(id_, lat, lon) -> dict:
    """Create canonical cone record dictionary."""
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


def load_or_create_cones() -> list:
    """Load cones from Excel and optionally merge with existing failures."""
    df = pd.read_excel(VENT_COORD)
    required = {"ID", "Latitude", "Longitude"}
    if not required.issubset(df.columns):
        raise ValueError(f"Excel must contain columns: {required}")

    # Create canonical cone records
    cones = [make_cone_record(r.ID, r.Latitude, r.Longitude) for r in df.itertuples()]

    # If failure CSV exists, update cones with previous state
    if FAILURE_LOG.exists():
        old_failures = pd.read_csv(FAILURE_LOG)
        old_dict = {row["id"]: row for idx, row in old_failures.iterrows()}

        # Update existing cones with saved state
        for cone in cones:
            if cone["id"] in old_dict:
                saved = old_dict[cone["id"]]
                cone.update({
                    "status": saved.get("status", cone["status"]),
                    "attempts": int(saved.get("attempts", 0)),
                    "error_type": saved.get("error_type"),
                    "error_msg": saved.get("error_msg"),
                    "traceback": saved.get("traceback"),
                    "last_attempt": saved.get("last_attempt"),
                    "next_retry_after": saved.get("next_retry_after")
                })

    return cones


# --- Worker Function ---
def process_cone_once(cone: dict) -> dict:
    """
    Process a single cone once.
    Returns updated cone record.
    """
    cone = cone.copy()
    cone["attempts"] += 1
    cone["last_attempt"] = utc_now()

    try:
        # Run DEM segmentation
        dem = dem_segment(
            cone["lat"], cone["lon"], cone["id"],
            POLYGON_FOLDER, DEM_FOLDER, diag=False
        )
        cone_dem, cone_poly, crater_poly, WARNING, warning_reasons = dem

        # Compute metrics
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

    # Set next retry time using exponential backoff
    cone["next_retry_after"] = (
        datetime.now(timezone.utc) + timedelta(seconds=compute_backoff(cone["attempts"]))
    ).isoformat()
    return cone


# --- Phase 1 ---
def phase_one_parallel(cones):
    """Process new cones in parallel."""
    logger.info("PHASE 1 STARTED")
    failures = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {executor.submit(process_cone_once, c): c["id"] for c in cones}

        for future in tqdm(as_completed(future_map), total=len(future_map), desc="Phase 1"):
            res = future.result()
            if res["status"] == "SUCCESS":
                logger.info(f"Cone {res['id']} SUCCESS")
            else:
                failures.append(res)
                logger.warning(f"Cone {res['id']} -> {res['status']} ({res['error_type']})")

    logger.info("PHASE 1 COMPLETE")
    return failures


# --- Phase 2 ---
def phase_two_parallel(failures):
    """Retry transient failures in parallel until exhausted."""
    logger.info("PHASE 2 STARTED")
    active = failures

    while active:
        now = datetime.now(timezone.utc)

        # Select cones eligible for retry
        retry_batch = [
            c for c in active
            if c["status"] == "RETRY_LATER"
            and c["attempts"] < MAX_TOTAL_ATTEMPTS
            and datetime.fromisoformat(c["next_retry_after"]) <= now
        ]

        # Cones that are still waiting or exhausted
        pending = [c for c in active if c not in retry_batch]

        if not retry_batch:
            break

        logger.info(f"Retrying {len(retry_batch)} cones")

        next_round = []
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_map = {executor.submit(process_cone_once, c): c["id"] for c in retry_batch}
            for future in tqdm(as_completed(future_map), total=len(future_map), desc="Phase 2"):
                res = future.result()
                if res["status"] == "SUCCESS":
                    logger.info(f"Cone {res['id']} RECOVERED")
                else:
                    if res["attempts"] >= MAX_TOTAL_ATTEMPTS:
                        res["status"] = "FAILED_FATAL"
                    next_round.append(res)

        active = pending + next_round

    logger.info("PHASE 2 COMPLETE")
    return active


# --- Main Driver ---
if __name__ == "__main__":
    start = time.perf_counter()
    print("Starting parallel two-phase cone processing pipeline...")

    cones = load_or_create_cones()

    # Phase 1: process new cones only
    new_cones = [c for c in cones if c["status"] in ("PENDING", "RETRY_LATER")]
    failures = phase_one_parallel(new_cones)

    # Persist failures after Phase 1
    if failures:
        pd.DataFrame(failures).to_csv(FAILURE_LOG, index=False)
        # Phase 2: retry transient failures
        final_failures = phase_two_parallel(failures)
        pd.DataFrame(final_failures).to_csv(FAILURE_LOG, index=False)

    runtime = time.perf_counter() - start
    logger.info(f"PIPELINE COMPLETE | Runtime {runtime:.2f}s")
    print(f"PIPELINE COMPLETE ({runtime:.2f}s)")
    print(f"Run log: {RUN_LOG}")
    print(f"Failure log: {FAILURE_LOG}")
