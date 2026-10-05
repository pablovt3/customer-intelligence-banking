import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / 'data' / 'raw'


def profile_table(file_name):
    df = pd.read_csv(
        RAW_DATA_DIR / file_name,
        sep=';'
    )

    print(f'\n{"=" * 60}')
    print(f'TABLE: {file_name}')
    print(f'{"=" * 60}')

    print('\nShape:')
    print(df.shape)

    print('\nData types:')
    print(df.dtypes)

    print('\nMissing values:')
    print(df.isna().sum())

    print('\nUnique values:')
    print(df.nunique())

    print('\nFirst rows:')
    print(df.head())

    for column in df.select_dtypes(include='str').columns:
        print(f'\nValue counts - {column}:')
        print(df[column].value_counts(dropna=False).head(20))


for file_path in sorted(RAW_DATA_DIR.glob('*.asc')):
    profile_table(file_path.name)


district = pd.read_csv(
    RAW_DATA_DIR / 'district.asc',
    sep=';'
)

print('\nDISTRICT A12 values:')
print(district['A12'].unique())

print('\nDISTRICT A15 values:')
print(district['A15'].unique())