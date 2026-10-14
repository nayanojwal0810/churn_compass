# Input data

The canonical IBM Telco Customer Churn sample is downloaded automatically on the first pipeline run if `telco_customer_churn.csv` is absent. The source URL and expected SHA-256 are defined in `src/churn_compass/data.py`; the downloaded bytes are checksum-verified before the file is written.

To use a local copy, place the CSV at `data/raw/telco_customer_churn.csv`. The pipeline checks its schema and data quality before modelling. Raw CSV files are intentionally not versioned in this repository; confirm the source's data-use terms before redistributing the data.
