import pandas as pd

from pathlib import Path
from sqlalchemy import text

from src.database import engine


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / 'data' / 'raw'


# ============================================================
# CATEGORY MAPPINGS
# ============================================================

FREQUENCY_MAPPING = {
    'POPLATEK MESICNE': 'MONTHLY',
    'POPLATEK TYDNE': 'WEEKLY',
    'POPLATEK PO OBRATU': 'AFTER_TRANSACTION'
}


STANDING_ORDER_K_SYMBOL_MAPPING = {
    'POJISTNE': 'INSURANCE_PAYMENT',
    'SIPO': 'HOUSEHOLD_PAYMENT',
    'LEASING': 'LEASING_PAYMENT',
    'UVER': 'LOAN_PAYMENT'
}


TRANSACTION_TYPE_MAPPING = {
    'PRIJEM': 'CREDIT',
    'VYDAJ': 'DEBIT',
    'VYBER': 'WITHDRAWAL'
}


TRANSACTION_OPERATION_MAPPING = {
    'VKLAD': 'CASH_DEPOSIT',
    'VYBER': 'CASH_WITHDRAWAL',
    'VYBER KARTOU': 'CARD_WITHDRAWAL',
    'PREVOD Z UCTU': 'TRANSFER_IN',
    'PREVOD NA UCET': 'TRANSFER_OUT'
}


TRANSACTION_K_SYMBOL_MAPPING = {
    'POJISTNE': 'INSURANCE_PAYMENT',
    'SLUZBY': 'STATEMENT_PAYMENT',
    'UROK': 'INTEREST_CREDITED',
    'SANKC. UROK': 'PENALTY_INTEREST',
    'SIPO': 'HOUSEHOLD_PAYMENT',
    'DUCHOD': 'PENSION',
    'UVER': 'LOAN_PAYMENT'
}


# ============================================================
# DATABASE
# ============================================================

def test_connection():
    with engine.connect():
        print('PostgreSQL connection successful.')


def get_table_count(table_name):
    query = text(
        f'SELECT COUNT(*) FROM {table_name}'
    )

    with engine.connect() as connection:
        result = connection.execute(query)

        return result.scalar()


# ============================================================
# EXTRACT
# ============================================================

def extract_table(file_name, **kwargs):
    return pd.read_csv(
        RAW_DATA_DIR / file_name,
        sep=';',
        **kwargs
    )


# ============================================================
# TRANSFORMATION UTILITIES
# ============================================================

def clean_text_column(series):
    return (
        series
        .str.strip()
        .replace('', pd.NA)
    )


def map_categories(
    series,
    mapping,
    column_name
):
    non_null_values = set(
        series.dropna().unique()
    )

    expected_values = set(
        mapping.keys()
    )

    unexpected_values = (
        non_null_values - expected_values
    )

    if unexpected_values:
        raise ValueError(
            f'Unexpected categories in {column_name}: '
            f'{sorted(unexpected_values)}'
        )

    return series.map(mapping)


# ============================================================
# TRANSFORM
# ============================================================

def transform_district(df):
    column_mapping = {
        'A1': 'district_id',
        'A2': 'district_name',
        'A3': 'region',
        'A4': 'inhabitants',
        'A5': 'municipalities_lt_499',
        'A6': 'municipalities_500_1999',
        'A7': 'municipalities_2000_9999',
        'A8': 'municipalities_gt_10000',
        'A9': 'cities',
        'A10': 'urban_ratio',
        'A11': 'average_salary',
        'A12': 'unemployment_rate_95',
        'A13': 'unemployment_rate_96',
        'A14': 'entrepreneurs_per_1000',
        'A15': 'crimes_95',
        'A16': 'crimes_96'
    }

    df = df.rename(
        columns=column_mapping
    )

    return df


def transform_account(df):
    df['date'] = pd.to_datetime(
        df['date'].astype(str),
        format='%y%m%d'
    )

    df['frequency'] = clean_text_column(
        df['frequency']
    )

    df['frequency'] = map_categories(
        df['frequency'],
        FREQUENCY_MAPPING,
        'account.frequency'
    )

    return df


def transform_client(df, age_reference_date):
    """Decode Berka demographics; retain the original birth_number unchanged.

    Source: docs/Financial Data description.pdf, p. 3: YYMMDD for men,
    YY(MM+50)DD for women. The 1900 century is an explicit assumption for
    this historical extract, consistent with the existing EDA.
    age is completed years at MAX(trans.asc.date), not current age or age
    at loan origination. Store the reference date so the snapshot is explicit.
    """
    df = df.copy()
    encoded = df['birth_number']
    if encoded.isna().any() or not encoded.between(0, 999999).all() or (encoded % 1 != 0).any():
        raise ValueError('birth_number must contain six-digit historical date codes.')
    month = encoded.floordiv(100).mod(100)
    if not (month.between(1, 12) | month.between(51, 62)).all():
        raise ValueError('Invalid Berka encoded birth month.')
    df['gender'] = month.gt(50).map({True: 'Female', False: 'Male'})
    df['birth_date'] = pd.to_datetime(dict(
        year=1900 + encoded.floordiv(10000),
        month=month.where(month.le(12), month - 50),
        day=encoded.mod(100)
    ), errors='raise')
    reference = pd.Timestamp(age_reference_date).normalize()
    if pd.isna(reference):
        raise ValueError('A valid historical age reference date is required.')
    df['age_reference_date'] = reference
    df['age'] = (reference.year - df['birth_date'].dt.year - (
        df['birth_date'].dt.month * 100 + df['birth_date'].dt.day
        > reference.month * 100 + reference.day
    ).astype(int))
    if not df['age'].between(0, 110).all():
        raise ValueError('Client age outside the historical plausibility range 0–110.')
    return df


def transform_disp(df):
    return df


def transform_card(df):
    df['issued'] = pd.to_datetime(
        df['issued'].str[:6],
        format='%y%m%d'
    )

    return df


def transform_standing_order(df):
    df['k_symbol'] = clean_text_column(
        df['k_symbol']
    )

    df['k_symbol'] = map_categories(
        df['k_symbol'],
        STANDING_ORDER_K_SYMBOL_MAPPING,
        'standing_order.k_symbol'
    )

    return df


def transform_loan(df):
    df['date'] = pd.to_datetime(
        df['date'].astype(str),
        format='%y%m%d'
    )

    return df


def transform_bank_transaction(df):
    df['date'] = pd.to_datetime(
        df['date'].astype(str),
        format='%y%m%d'
    )

    df = df.rename(
        columns={
            'bank': 'counterparty_bank',
            'account': 'counterparty_account'
        }
    )

    text_columns = [
        'type',
        'operation',
        'k_symbol',
        'counterparty_bank'
    ]

    for column in text_columns:
        df[column] = clean_text_column(
            df[column]
        )

    df['type'] = map_categories(
        df['type'],
        TRANSACTION_TYPE_MAPPING,
        'bank_transaction.type'
    )

    df['operation'] = map_categories(
        df['operation'],
        TRANSACTION_OPERATION_MAPPING,
        'bank_transaction.operation'
    )

    df['k_symbol'] = map_categories(
        df['k_symbol'],
        TRANSACTION_K_SYMBOL_MAPPING,
        'bank_transaction.k_symbol'
    )

    df['counterparty_account'] = (
        df['counterparty_account']
        .astype('Int64')
    )

    return df


# ============================================================
# LOAD
# ============================================================

def load_table(
    df,
    table_name,
    chunksize=None
):
    df.to_sql(
        name=table_name,
        con=engine,
        if_exists='append',
        index=False,
        chunksize=chunksize
    )

    print(
        f'Loaded {len(df):,} rows into {table_name}.'
    )


# ============================================================
# VALIDATION
# ============================================================

def validate_table(
    table_name,
    expected_rows
):
    actual_rows = get_table_count(
        table_name
    )

    if actual_rows != expected_rows:
        raise ValueError(
            f'Validation failed for {table_name}: '
            f'expected {expected_rows:,} rows, '
            f'found {actual_rows:,}.'
        )

    print(
        f'Validated {table_name}: '
        f'{actual_rows:,} rows.'
    )


# ============================================================
# PIPELINE
# ============================================================

def run_pipeline():
    print('\nStarting ETL pipeline...\n')

    # Derive the snapshot from the same source that will be loaded below.
    bank_transaction = transform_bank_transaction(
        extract_table('trans.asc', low_memory=False)
    )
    age_reference_date = bank_transaction['date'].max()

    # --------------------------------------------------------
    # DISTRICT
    # --------------------------------------------------------

    district = extract_table(
        'district.asc',
        na_values='?'
    )

    district = transform_district(
        district
    )

    load_table(
        district,
        'district'
    )

    validate_table(
        'district',
        len(district)
    )

    # --------------------------------------------------------
    # ACCOUNT
    # --------------------------------------------------------

    account = extract_table(
        'account.asc'
    )

    account = transform_account(
        account
    )

    load_table(
        account,
        'account'
    )

    validate_table(
        'account',
        len(account)
    )

    # --------------------------------------------------------
    # CLIENT
    # --------------------------------------------------------

    client = extract_table(
        'client.asc'
    )

    client = transform_client(
        client, age_reference_date
    )

    load_table(
        client,
        'client'
    )

    validate_table(
        'client',
        len(client)
    )

    # --------------------------------------------------------
    # DISPOSITION
    # --------------------------------------------------------

    disp = extract_table(
        'disp.asc'
    )

    disp = transform_disp(
        disp
    )

    load_table(
        disp,
        'disp'
    )

    validate_table(
        'disp',
        len(disp)
    )

    # --------------------------------------------------------
    # CARD
    # --------------------------------------------------------

    card = extract_table(
        'card.asc'
    )

    card = transform_card(
        card
    )

    load_table(
        card,
        'card'
    )

    validate_table(
        'card',
        len(card)
    )

    # --------------------------------------------------------
    # STANDING ORDER
    # --------------------------------------------------------

    standing_order = extract_table(
        'order.asc'
    )

    standing_order = (
        transform_standing_order(
            standing_order
        )
    )

    load_table(
        standing_order,
        'standing_order'
    )

    validate_table(
        'standing_order',
        len(standing_order)
    )

    # --------------------------------------------------------
    # LOAN
    # --------------------------------------------------------

    loan = extract_table(
        'loan.asc'
    )

    loan = transform_loan(
        loan
    )

    load_table(
        loan,
        'loan'
    )

    validate_table(
        'loan',
        len(loan)
    )

    # --------------------------------------------------------
    # BANK TRANSACTION
    # --------------------------------------------------------

    load_table(
        bank_transaction,
        'bank_transaction',
        chunksize=10000
    )

    validate_table(
        'bank_transaction',
        len(bank_transaction)
    )

    print(
        '\nETL pipeline completed successfully.'
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    test_connection()
    run_pipeline()