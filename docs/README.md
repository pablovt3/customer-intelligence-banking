# The Berka Dataset

This is the Berka Dataset, originally from The Discovery Challenge, held as a part of the 3rd European Conference on Principles and Practice of Knowledge Discovery in Databases (PKDD’99), September 15 - 18, 1999, Prague, Czech Republic.


## Reproducible demographic transformation

`src/etl.py::transform_client` owns the decoding of `client.birth_number`.
The local Financial Data description.pdf (page 3) specifies YYMMDD for men
and YY(MM+50)DD for women. Original numeric codes are retained unchanged.
The century is explicitly assumed to be 1900 for this historical extract,
matching the original EDA; this is not a general Czech national-ID decoder.
Invalid encoded months or calendar dates fail the load rather than silently
becoming missing values. `gender` records source-encoded sex (Female/Male).

`age` is completed years at `age_reference_date`, the maximum transaction
date in the source being loaded (1998-12-31 for this extract). The ETL derives
this date from trans.asc before loading clients and stores it on every client.
Age is a snapshot attribute, not a time-invariant fact. Rebuild it when the
extract changes. For future loan prediction, calculate age at origination
from birth_date and the loan date; do not substitute this end-of-extract age.
Both EDA notebooks read demographics from PostgreSQL and assert that the
stored reference matches the observed transaction cutoff. Their date-of-birth
versus account-opening check does not create a second demographic age field.

From the project root, with the customer-intelligence environment activated
and the existing .env database connection configured, rebuild as follows.
This drops and recreates the eight source tables:

```sh
python -c 'from pathlib import Path; from src.database import engine; connection = engine.connect(); transaction = connection.begin(); connection.exec_driver_sql(Path("database/schema.sql").read_text()); transaction.commit(); connection.close()'
python -m src.etl
```

Run both notebooks from their notebooks directory with the same environment.
The load verifies all source row counts. Database constraints enforce non-null
demographics, the two encoded categories, age 0–110, birth before the reference,
and consistency of completed age with the stored dates. The original
transaction 236564 and account 9051 interest records are not corrected or removed.
