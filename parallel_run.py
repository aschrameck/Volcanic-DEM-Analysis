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
import requests
import pandas as pd
from pathlib import Path
from requests.exceptions import Timeout
from rasterio.errors import RasterioIOError
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager
from tqdm import tqdm
import os

from adaptive_dem_segment import dem_segment, NullError, DownloadError, DiskSpaceError, DEMSizeError
from measure import cone_metrics, CRS_Error
from basal_surface import BasalSurfaceError

# --- Configuration ---
# File paths
POLYGON_FOLDER = Path(r"D:\Polygons")
DEM_FOLDER = Path(r"D:\DEMs")
VENT_COORD = Path(r"D:\vent_coords.xls")
CSV_OUT = Path(r"D:\Metrics.csv")

RUN_LOG = Path(r"D:\cone_run.log")
FAILURE_LOG = Path(r"D:\cone_failures.csv")

# Retry configuration
BASE_RETRY_DELAY = 30       # seconds
MAX_RETRY_DELAY = 300       # seconds
MAX_TOTAL_ATTEMPTS = 3      # maximum attempts per cone

MAX_WORKERS = max(1, os.cpu_count() - 3)  # leave 3 CPUs free

# Error classification
TRANSIENT_ERRORS = (DownloadError, Timeout, TimeoutError, RasterioIOError)
FATAL_ERRORS = (NullError, DiskSpaceError, CRS_Error, BasalSurfaceError, DEMSizeError)

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

# TNM Access API service health check
CHECK_INTERVAL = 120        # seconds between checks
MAX_OUTAGE_WAIT = 900              # max total wait (15 min)


# --- Helper Functions ---
def compute_backoff(attempts: int) -> int:
    """Compute exponential backoff (seconds) with cap."""
    return min(BASE_RETRY_DELAY * (2 ** (attempts - 1)), MAX_RETRY_DELAY)


def tnm_health_check():
    """Check if TNM Access API is responding correctly."""
    url = "https://tnmaccess.nationalmap.gov/api/v1/products"
    params = {
        "bbox": "-111.6,35.5,-111.5,35.6",
        "datasets": "National Elevation Dataset (NED) 1/3 arc-second",
        "prodFormats": "GeoTIFF"
    }

    try:
        r = requests.get(url, params=params, timeout=60)

        if r.status_code != 200:
            return False

        # Check JSON is actually parseable
        r.json()
        return True

    except Exception:
        return False


def wait_for_services(outage_start, last_check, pipeline_abort):
    """Wait if external services are down, with rate-limited checks."""
    now = time.time()

    # Rate-limit status checks
    if now - last_check.value < CHECK_INTERVAL:
        return

    last_check.value = now

    if tnm_health_check():
        if outage_start.value > 0:
            logger.info("External services recovered")
        outage_start.value = 0
        return

    # Services are DOWN
    if outage_start.value == 0:
        outage_start.value = now
        logger.error("External services DOWN — entering outage wait")

    waited = now - outage_start.value
    if waited >= MAX_OUTAGE_WAIT:
        logger.critical(
            f"External services down for {waited:.0f}s "
            f"(max {MAX_OUTAGE_WAIT}s) — aborting run safely"
        )
        pipeline_abort.value = True
        return

    logger.warning("Services down — deferring work to main loop")
    return


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
def process_cone_once(cone, lock, outage_start, last_check, pipeline_abort) -> dict:
    """Process a single cone once and safely write metrics with lock."""
    cone = cone.copy()
    cone["attempts"] += 1
    cone["last_attempt"] = time.time()

    try:
        # Check external services status
        wait_for_services(
            outage_start=outage_start,
            last_check=last_check,
            pipeline_abort=pipeline_abort
        )

        # If a global outage was declared, exit safely
        if pipeline_abort.value:
            cone["status"] = "RETRY_OUTAGE"
            cone["error_type"] = "ExternalServiceOutage"
            cone["error_msg"] = "External services unavailable"
            cone["traceback"] = None
            cone["next_retry_after"] = None
            return cone

    # Process the cone
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
        time.time() + compute_backoff(cone["attempts"])
    )
    return cone


# --- Phase 1 ---
def phase_one_parallel(cones, lock, dl_error_count, cooldown_until, pipeline_abort):
    """Process new cones in parallel."""
    logger.info("PHASE 1 STARTED")
    failures = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all cones for processing
        future_map = {executor.submit(process_cone_once, c, lock, OUTAGE_START, LAST_UPTIME_CHECK, pipeline_abort):
                      c["id"] for c in cones}

        # Collect results as they complete
        for future in tqdm(as_completed(future_map), total=len(future_map), desc="Phase 1"):
            # Check for global abort
            if pipeline_abort.value:
                logger.critical("Abort flag set — stopping Phase 1 collection")
                break

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

                if res["error_type"] == "ExternalServiceOutage":
                    failures.append(res)
                    pipeline_abort.value = True
                    logger.critical("Global outage detected — stopping Phase 1")
                    break

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
                                     f"Entering cooldown for {COOLDOWN_SECONDS}s")
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
def phase_two_parallel(failures, lock, dl_error_count, cooldown_until, pipeline_abort):
    """
    Retry transient failures in parallel until exhausted.
    """
    logger.info("PHASE 2 STARTED")

    # Things still eligible for retry
    retry_queue = [c for c in failures if c["status"] == "RETRY_LATER"]

    # Permanent failures (fatal or exhausted retries)
    final_failures = [c for c in failures if c["status"] == "FAILED_FATAL"]

    while retry_queue:
        # Check for global abort
        if pipeline_abort.value:
            logger.critical("Abort flag set — stopping Phase 2")
            break

        now = time.time()

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
            and c["next_retry_after"] is not None
            and c["next_retry_after"] <= now
        ]

        waiting = [c for c in retry_queue if (c not in ready and c["attempts"] < MAX_TOTAL_ATTEMPTS)]

        if not ready:
            # Sleep until the next retry time
            if waiting:
                next_times = [c["next_retry_after"] for c in waiting if c["next_retry_after"]]
                sleep_time = max(0, min(next_times) - time.time())
                sleep_time = min(sleep_time, 60)  # cap sleep

                logger.info(f"Sleeping {sleep_time:.1f}s until next retry")

                for _ in range(int(sleep_time)):
                    if pipeline_abort.value:
                        logger.critical("Abort detected during retry sleep")
                        return final_failures + retry_queue
                    time.sleep(1)
            retry_queue = waiting
            continue

        logger.info(f"Retrying {len(ready)} cones")

        next_retry_queue = []

        # Process ready cones in parallel
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(process_cone_once, c, lock, OUTAGE_START, LAST_UPTIME_CHECK, PIPELINE_ABORT): c["id"]
                for c in ready}

            for future in tqdm(as_completed(futures), total=len(futures), desc="Phase 2"):
                try:
                    res = future.result()

                    if res["error_type"] == "ExternalServiceOutage":
                        failures.append(res)
                        PIPELINE_ABORT.value = True
                        logger.critical("Global outage detected — stopping Phase 2")
                        break

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
                                         f"Entering cooldown for {COOLDOWN_SECONDS}s")
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

    # Global download error tracking
    DOWNLOAD_ERROR_COUNT = manager.Value("i", 0)
    COOLDOWN_UNTIL = manager.Value("d", 0.0)

    # TNM API Uptime monitoring
    OUTAGE_START = manager.Value("d", 0.0)
    LAST_UPTIME_CHECK = manager.Value("d", 0.0)
    PIPELINE_ABORT = manager.Value("b", False)

    # Begin processing pipeline
    start = time.perf_counter()
    print("Starting parallel two-phase cone processing pipeline...")

    cones = load_or_create_cones()

    # Phase 1: process new cones only (skip SUCCESS)
    new_cones = [c for c in cones if c["status"] == "PENDING"]

    failures = phase_one_parallel(
        new_cones,
        METRICS_LOCK,
        DOWNLOAD_ERROR_COUNT,
        COOLDOWN_UNTIL,
        PIPELINE_ABORT
    )

    if failures:
        # Save intermediate failures
        pd.DataFrame(failures).to_csv(FAILURE_LOG, index=False)

        # Phase 2: retry transient failures
        final_failures = phase_two_parallel(
            failures,
            METRICS_LOCK,
            DOWNLOAD_ERROR_COUNT,
            COOLDOWN_UNTIL,
            PIPELINE_ABORT
        )
        pd.DataFrame(final_failures).to_csv(FAILURE_LOG, index=False)

    if PIPELINE_ABORT.value:
        logger.critical("Pipeline halted due to prolonged external outage")
        print("Pipeline halted due to external outage. Safe to resume later.")
        raise SystemExit(2)

    runtime = time.perf_counter() - start
    logger.info(f"PIPELINE COMPLETE | Runtime {runtime:.2f}s")
    print(f"PIPELINE COMPLETE ({runtime:.2f}s)")
    print(f"Run log: {RUN_LOG}")
    print(f"Failure log: {FAILURE_LOG}")
