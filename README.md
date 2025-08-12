# Galaxy-Global-Scaling-Laws

Code and analysis for the paper 'Discovery of Non-Linear Morphology-Energy Coupling in Galaxy Dynamics'



\# Discovery of Non-Linear Morphology-Energy Coupling in Galaxy Dynamics



This repository contains the full analysis code, data, and article source for the paper: "Resolving Inconsistencies in Galaxy Dynamics through Environment-Dependent Gravity."



\## Abstract



We resolve systematic inconsistencies in SPARC galaxy data through physical corrections, establishing Global Scaling Laws (GSL) as a robust framework. By unifying environment-dependent gravity with angular momentum-regulated interactions, we demonstrate: (1) Strong correlation between morphology and binding energy ($r = -0.84$, $p < 10^{-48}$); (2) A "Unified Interaction" GSL model achieves \*\*93.9% success\*\* ($\\chi^2\_{\\rm red} = 0.25$) vs. $\\Lambda$CDM (62.6%); (3) The theory is supported by a UV fixed point at $(c\_1^\*, c\_2^\*, \\alpha\_g^\*) = (0.82, -0.38, 0.29)$ and is consistent with cosmological parameters ($H\_0$, $\\sigma\_8$). Solar-system safety is demonstrated via a PPN screening mechanism.



\## Repository Structure



\-   \*\*/paper\*\*: Contains the LaTeX source (`main.tex` and `references.bib`) for the research article.

\-   \*\*/scripts\*\*: Contains all Python scripts used for the analysis.

\-   \*\*/data\*\*: Contains the necessary metadata file (`SPARC\_Lelli2016\_masses\_cleaned.csv`).

\-   \*\*/results\*\*: Contains the key outputs of the analysis, including the final model comparison table and figures.

\-   `requirements.txt`: A list of required Python libraries.



\## Scripts Overview



\* `scripts/full\_comparison.py`: The main experimental script. It loads the data, performs a robust train/test split, and runs a head-to-head comparison of all competing paradigms to produce the final performance leaderboard.

\* `scripts/analyze\_failure.py`: Performs a deep-dive diagnostic analysis on the galaxies for which the winning model failed, identifying systematic patterns and testing key theoretical predictions.

\* `scripts/theory.py`: A script that outputs a summary of the theoretical framework.

\* `scripts/predictions.py`: A script that outputs the specific, falsifiable predictions of the final theory.

\* `scripts/paper\_figures.py`: A script to generate all figures used in the final paper.



\## Setup and Reproduction



To reproduce the results of this study, please follow these steps:



1\.  \*\*Clone this repository:\*\*

&nbsp;   ```bash

&nbsp;   git clone \[https://github.com/](https://github.com/)KarmirisP/Galaxy-Global-Scaling-Laws.git

&nbsp;   cd Galaxy-Global-Scaling-Laws

&nbsp;   ```



2\.  \*\*Create a Python environment and install dependencies:\*\*

&nbsp;   ```bash

&nbsp;   pip install -r requirements.txt

&nbsp;   ```



3\.  \*\*Download the SPARC Data:\*\* The raw galaxy rotation curve data is publicly available from the \*\*\[official SPARC website](http://astroweb.cwru.edu/SPARC/)\*\*. Please download the `sparc\_data` folder and place it in the main directory of this repository.



4\.  \*\*Run the analysis:\*\* Execute the main comparison script from the terminal.

&nbsp;   ```bash

&nbsp;   python scripts/full\_comparison.py

&nbsp;   ```

This will run the full train/test analysis for all models and generate the final results table in the `results/` directory.



\## Citation



If you use this code or its results in your research, please cite the accompanying paper and this repository.





