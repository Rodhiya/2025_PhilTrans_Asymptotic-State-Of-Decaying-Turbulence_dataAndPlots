# Replication Data and Scripts for *The Asymptotic State of Decaying Turbulence*

This repository contains the processed datasets, analysis scripts, and figure-generation code needed to reproduce the results in:

- **Title:** *The Asymptotic State of Decaying Turbulence*
- **Authors:** Akash Rodhiya, Katepalli R. Sreenivasan
- **Journal:** *Philosophical Transactions A*
- **Year:** 2025
- **DOI:** (to be added)

---

## Repository Structure

```text
.
├── data/               # Processed binary data files used for plotting (.npy)
├── figures/            # Output directory for generated figures
├── scripts/            # Analysis and figure-generation scripts (Python)
└── requirements.txt    # List all the dependencies
└── README.md           # This file

```

## Prerequisites
To run the scripts in this repository, you will need Python 3

### Python Dependencies
It is recommended to create a virtual environment. The core dependencies will be listed in requirements.txt

## Usage
To regenerate the figures found in the manuscript, follow these steps:

1. Clone the repository
   
git clone https://github.com/Rodhiya/2025_PhilTrans_Asymptotic-State-Of-Decaying-Turbulence_dataAndPlots.git;
cd 2025_PhilTrans_Asymptotic-State-Of-Decaying-Turbulence_dataAndPlots

2. Install dependencies: pip install -r requirements.txt

3. Generate Figures: Scripts are named according to the Figure number from the paper.

4. Output: The resulting plots will be saved in the figures/ directory.

## Data Description
The data/ folder contains processed statistics derived from direct numerical simulations (DNS) of decaying turbulence.

- Note: Due to size constraints, the raw 3D flow fields are not included here.
- File Format: Data is stored in binary format (.npy)
- If you require access to the full raw simulation data, please contact us via email.

## Contact
For any questions regarding the code or data, please contact: Akash Rodhiya - akashrodhiya10[at].gmail.com
