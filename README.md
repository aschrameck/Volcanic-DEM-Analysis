# Automated Volcanic Field DEM Analysis (NASA Space Grant Project)

**GitHub Link:** [Volcanic-DEM-Analysis](https://github.com/aschrameck/Volcanic-DEM-Analysis)

---

## Project Overview

This project automates the analysis of cinder cone digital elevation models (DEMs) to quantify volcanic landform morphology and perform cluster analysis on extracted features.

Key objectives include:

* Automatically segmenting cinder cone DEMs.
* Extracting cone and crater measurements such as width, elevation, basal elevation, rim height, ellipticity, elongation, circularity, orientation, area, and crater depth.
* Performing extraction in parallel to decrease processing time.
* Apply workflow to the San Francisco Volcanic Field. 

---

## Project Structure

```
Volcanic-DEM-Analysis/
|
├── DEMs/                    # Folder to store DEM downloads
├── Polygons/                # Folder to store cone and crater polygons
├── data/                    # Example cone segmenting and visualizations
├── output/                  # Output folder for metrics and logs
├── basal_surface.py         # Functions to interpolate the basal surface
├── cone-metrics.py          # Computes metrics from cone DEMs
├── dem-segment.py           # Core DEM segmentation functions
├── parallel_run.py          # Pipeline with parallelization
├── radial-segment.py        # OBSOLETE - Radial segmentation for base and crater boundaries
├── run.py                   # Serial (single-threaded) pipeline
├── vent_coords.xls          # Example vent coordinate input
├── poster.pdf               # Research poster for project
├── presentation.pptx        # Slideshow presentation for project
├── README.md
```

## Set Up

### Step 1: Clone the Repository

```bash
git clone https://github.com/aschrameck/Volcanic-DEM-Analysis.git
cd Volcanic-DEM-Analysis
```

### Step 2: Set Up Python Environment

* Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

* Ensure Python 3.9+ is installed.

Install required packages:
```bash
pip install pandas tqdm requests rasterio shapely
```

### Step 3: Running Python Scripts

Update paths in run.py or parallel_run.py for:
* POLYGON_FOLDER
* DEM_FOLDER
* VENT_COORD
* CSV_OUT

#### Serial execution (single-threaded)
```bash
python run.py
```
* Suitable for small batches or debugging.
* Generates cone_run.log and cone_failures.csv.
* Handles Phase 1 and Phase 2 sequentially.

#### Parallel execution
```bash
python parallel_run.py
```
* Uses multiple processes for faster processing.
* Supports progress bars for both phases.
* Automatically resumes from cone_failures.csv if the pipeline is interrupted.
* Generates updated cone_failures.csv and run log.

---

## Logging

* Run log: cone_run.log — contains detailed information for each cone and retry attempt.
* Failure log: cone_failures.csv — tracks all cones that failed, with retry counters and timestamps.

## License

This project is released under the MIT License.
