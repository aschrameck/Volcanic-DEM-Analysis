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
9. Global download-error cooldown to avoid hammering API.
"""

import time
import traceback
import logging
import pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path
from requests.exceptions import Timeout
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager
from tqdm import tqdm
import os

from adaptive_dem_segment import dem_segment, NullError, DownloadError, DiskSpaceError
from measure import cone_metrics, CRS_Error

# --- Configuration ---
# File paths
POLYGON_FOLDER = Path(r"D:\Polygons")
DEM_FOLDER = Path(r"D:\DEMs")
VENT_COORD = Path(r"D:\vent_coords.xls")
CSV_OUT = Path(r"D:\metrics.csv")

RUN_LOG = Path(r"D:\cone_run.log")
FAILURE_LOG = Path(r"D:\cone_failures.csv")

# Retry configuration
BASE_RETRY_DELAY = 30       # seconds
MAX_RETRY_DELAY = 300       # seconds
MAX_TOTAL_ATTEMPTS = 3      # maximum attempts per cone

MAX_WORKERS = max(1, os.cpu_count() - 3)  # leave 3 CPUs free

# Error classification
TRANSIENT_ERRORS = (DownloadError, Timeout, TimeoutError)
FATAL_ERRORS = (NullError, DiskSpaceError, CRS_Error)

# Download cooldown configuration
DOWNLOAD_ERROR_THRESHOLD = 3   # consecutive DownloadErrors triggers cooldown
COOLDOWN_SECONDS = 300          # cooldown duration in seconds

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
    # Read in vent coordinates
    df = pd.read_excel(VENT_COORD)
    required = {"ID", "Latitude", "Longitude"}
    if not required.issubset(df.columns):
        raise ValueError(f"Excel must contain columns: {required}")

    # Create initial cone records
    cones = [make_cone_record(r.ID, r.Latitude, r.Longitude) for r in df.itertuples()]

    # Merge previous failures
    if FAILURE_LOG.exists():
        old_failures = pd.read_csv(FAILURE_LOG)
        old_dict = {row["id"]: row for idx, row in old_failures.iterrows()}
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
def process_cone_once(cone: dict, lock) -> dict:
    """Process a single cone once and safely write metrics with lock."""
    cone = cone.copy()
    cone["attempts"] += 1
    cone["last_attempt"] = utc_now()

    # Process the cone
    try:
        dem = dem_segment(
            cone["lat"], cone["lon"], cone["id"],
            POLYGON_FOLDER, DEM_FOLDER, diag=False
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

        # If it worked, set status to SUCCESS
        cone["status"] = "SUCCESS"
        logger.info(f"Cone {cone['id']} SUCCESS")
        return cone

    # Handle errors
    except FATAL_ERRORS as e:
        cone["status"] = "FAILED_FATAL"
        cone["error_type"] = type(e).__name__
        cone["error_msg"] = str(e)
        cone["traceback"] = traceback.format_exc()
        logger.warning(f"Cone {cone['id']} -> {cone['status']} (reason: {cone['error_type']})")
        return cone

    except TRANSIENT_ERRORS as e:
        cone["status"] = "RETRY_LATER"
        cone["error_type"] = type(e).__name__
        cone["error_msg"] = str(e)
        cone["traceback"] = traceback.format_exc()
        logger.warning(f"Cone {cone['id']} -> {cone['status']} (reason: {cone['error_type']})")

    except Exception as e:
        cone["status"] = "RETRY_LATER"
        cone["error_type"] = type(e).__name__
        cone["error_msg"] = str(e)
        cone["traceback"] = traceback.format_exc()
        logger.warning(f"Cone {cone['id']} -> {cone['status']} (reason: {cone['error_type']})")

    cone["next_retry_after"] = (
        datetime.now(timezone.utc) + timedelta(seconds=compute_backoff(cone["attempts"]))
    ).isoformat()
    return cone


# --- Phase 1 ---
def phase_one_parallel(cones, lock, dl_error_count, cooldown_until):
    """Process new cones in parallel."""
    logger.info("PHASE 1 STARTED")
    failures = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all cones for processing
        future_map = {executor.submit(process_cone_once, c, lock): c["id"] for c in cones}

        # Collect results as they complete
        for future in tqdm(as_completed(future_map), total=len(future_map), desc="Phase 1"):
            # Check global cooldown
            now = time.time()
            if cooldown_until.value > now:
                sleep_time = cooldown_until.value - now
                logger.warning(f"GLOBAL DOWNLOAD COOLDOWN ACTIVE ({sleep_time:.1f}s remaining)")
                time.sleep(sleep_time)
                logger.info("GLOBAL DOWNLOAD COOLDOWN ENDED")

            # Process result
            try:
                res = future.result()

                # If successful, reset download error count
                if res["status"] == "SUCCESS":
                    dl_error_count.value = 0

                # Handle DownloadError and track consecutive occurrences
                elif res["error_type"] == "DownloadError":
                    dl_error_count.value += 1
                    logger.warning(f"Cone {res['id']} DownloadError (consecutive={dl_error_count.value})")

                    # Trigger global cooldown if threshold exceeded
                    if dl_error_count.value >= DOWNLOAD_ERROR_THRESHOLD:
                        cooldown_until.value = time.time() + COOLDOWN_SECONDS
                        logger.error(f"DOWNLOAD ERROR THRESHOLD HIT ({DOWNLOAD_ERROR_THRESHOLD})."
                                     "Entering cooldown for {COOLDOWN_SECONDS}s")
                        dl_error_count.value = 0

                    failures.append(res)

                # Handle other failures
                elif res["status"] == "FAILED_FATAL":
                    dl_error_count.value = 0
                    failures.append(res)
                else:
                    dl_error_count.value = 0
                    failures.append(res)

            except Exception as e:
                logger.error(f"Worker crash: {e}")

    logger.info("PHASE 1 COMPLETE")
    return failures


# --- Phase 2 ---
def phase_two_parallel(failures, lock, dl_error_count, cooldown_until):
    """
    Retry transient failures in parallel until exhausted.
    """
    logger.info("PHASE 2 STARTED")

    # Things still eligible for retry
    retry_queue = [c for c in failures if c["status"] == "RETRY_LATER"]

    # Permanent failures (fatal or exhausted retries)
    final_failures = [c for c in failures if c["status"] == "FAILED_FATAL"]

    while retry_queue:
        now = datetime.now(timezone.utc)

        # Global cooldown check
        if cooldown_until.value > now:
            sleep_time = cooldown_until.value - now
            logger.warning(f"GLOBAL DOWNLOAD COOLDOWN ACTIVE ({sleep_time:.1f}s remaining)")
            time.sleep(sleep_time)
            logger.info("GLOBAL DOWNLOAD COOLDOWN ENDED")

        # Find cones ready for retry
        ready = [
            c for c in retry_queue
            if c["attempts"] < MAX_TOTAL_ATTEMPTS
            and datetime.fromisoformat(c["next_retry_after"]) <= now
        ]

        waiting = [c for c in retry_queue if c not in ready]

        if not ready:
            # Sleep until the next retry time
            if waiting:
                next_times = [
                    datetime.fromisoformat(c["next_retry_after"]).timestamp()
                    for c in waiting
                ]
                sleep_time = max(0, min(next_times) - time.time())
                logger.info(f"Sleeping {sleep_time:.1f}s until next retry")
                time.sleep(sleep_time)
            retry_queue = waiting
            continue

        logger.info(f"Retrying {len(ready)} cones")

        next_retry_queue = []

        # Process ready cones in parallel
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(process_cone_once, c, lock): c["id"]
                for c in ready
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc="Phase 2"):
                try:
                    res = future.result()

                    # If successful, reset download error count
                    if res["status"] == "SUCCESS":
                        logger.info(f"Cone {res['id']} RECOVERED")
                        dl_error_count.value = 0

                    # Handle DownloadError and track consecutive occurrences
                    elif res["error_type"] == "DownloadError":
                        dl_error_count.value += 1
                        logger.warning(f"Cone {res['id']} DownloadError (consecutive={dl_error_count.value})")

                        # Trigger global cooldown if threshold exceeded
                        if dl_error_count.value >= DOWNLOAD_ERROR_THRESHOLD:
                            cooldown_until.value = time.time() + COOLDOWN_SECONDS
                            logger.error(f"DOWNLOAD ERROR THRESHOLD HIT ({DOWNLOAD_ERROR_THRESHOLD})."
                                         "Entering cooldown for {COOLDOWN_SECONDS}s")
                            dl_error_count.value = 0

                        next_retry_queue.append(res)

                    # If total attempts exhausted, mark as fatal
                    elif res["attempts"] >= MAX_TOTAL_ATTEMPTS:
                        res["status"] = "FAILED_FATAL"
                        final_failures.append(res)
                        logger.warning(
                            f"Cone {res['id']} exhausted retries -> FAILED_FATAL"
                        )
                        dl_error_count.value = 0

                    # Handle other failures
                    else:
                        dl_error_count.value = 0
                        next_retry_queue.append(res)

                except Exception as e:
                    logger.error(f"Worker crash: {e}")

        retry_queue = waiting + next_retry_queue

    logger.info("PHASE 2 COMPLETE")

    # Return everything that is not SUCCESS
    return final_failures + retry_queue


# --- Main Driver ---
if __name__ == "__main__":
    # Set up multiprocessing shared variables
    manager = Manager()
    METRICS_LOCK = manager.Lock()

    DOWNLOAD_ERROR_COUNT = manager.Value("i", 0)
    COOLDOWN_UNTIL = manager.Value("d", 0.0)

    # Begin processing pipeline
    start = time.perf_counter()
    print("Starting parallel two-phase cone processing pipeline...")

    cones = load_or_create_cones()

    # Phase 1: process new cones only (skip SUCCESS)
    new_cones = [c for c in cones if c["status"] == "PENDING"]
    failures = phase_one_parallel(new_cones, METRICS_LOCK, DOWNLOAD_ERROR_COUNT, COOLDOWN_UNTIL)

    # Persist failures after Phase 1
    if failures:
        pd.DataFrame(failures).to_csv(FAILURE_LOG, index=False)

        # Phase 2: retry transient failures
        final_failures = phase_two_parallel(failures, METRICS_LOCK, DOWNLOAD_ERROR_COUNT, COOLDOWN_UNTIL)
        pd.DataFrame(final_failures).to_csv(FAILURE_LOG, index=False)

    runtime = time.perf_counter() - start
    logger.info(f"PIPELINE COMPLETE | Runtime {runtime:.2f}s")
    print(f"PIPELINE COMPLETE ({runtime:.2f}s)")
    print(f"Run log: {RUN_LOG}")
    print(f"Failure log: {FAILURE_LOG}")
