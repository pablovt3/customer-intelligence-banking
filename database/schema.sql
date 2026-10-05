-- ============================================================
-- Customer Intelligence Suite - Retail Banking
-- Database schema
-- Source: Berka Dataset
-- ============================================================


-- ============================================================
-- DROP TABLES
-- Allows the schema to be recreated during development.
-- Tables are dropped in reverse dependency order.
-- ============================================================

DROP TABLE IF EXISTS bank_transaction CASCADE;
DROP TABLE IF EXISTS loan CASCADE;
DROP TABLE IF EXISTS standing_order CASCADE;
DROP TABLE IF EXISTS card CASCADE;
DROP TABLE IF EXISTS disp CASCADE;
DROP TABLE IF EXISTS client CASCADE;
DROP TABLE IF EXISTS account CASCADE;
DROP TABLE IF EXISTS district CASCADE;


-- ============================================================
-- DISTRICT
-- Demographic and socioeconomic information by district.
-- ============================================================

CREATE TABLE district (
    district_id INTEGER PRIMARY KEY,
    district_name TEXT NOT NULL,
    region TEXT NOT NULL,
    inhabitants INTEGER NOT NULL,
    municipalities_lt_499 INTEGER NOT NULL,
    municipalities_500_1999 INTEGER NOT NULL,
    municipalities_2000_9999 INTEGER NOT NULL,
    municipalities_gt_10000 INTEGER NOT NULL,
    cities INTEGER NOT NULL,
    urban_ratio NUMERIC,
    average_salary NUMERIC,
    unemployment_rate_95 NUMERIC,
    unemployment_rate_96 NUMERIC,
    entrepreneurs_per_1000 NUMERIC,
    crimes_95 INTEGER,
    crimes_96 INTEGER
);


-- ============================================================
-- ACCOUNT
-- Bank account information.
-- frequency values are standardized during the ETL process.
-- ============================================================

CREATE TABLE account (
    account_id INTEGER PRIMARY KEY,
    district_id INTEGER NOT NULL,
    date DATE NOT NULL,
    frequency TEXT NOT NULL,

    CONSTRAINT fk_account_district
        FOREIGN KEY (district_id)
        REFERENCES district(district_id),

    CONSTRAINT chk_account_frequency
        CHECK (
            frequency IN (
                'MONTHLY',
                'WEEKLY',
                'AFTER_TRANSACTION'
            )
        )
);


-- ============================================================
-- CLIENT
-- Customer information.
-- birth_number is kept in its original encoded form.
-- ============================================================

CREATE TABLE client (
    client_id INTEGER PRIMARY KEY,
    birth_number INTEGER NOT NULL,
    birth_date DATE NOT NULL,
    gender TEXT NOT NULL CHECK (gender IN ('Female', 'Male')),
    age SMALLINT NOT NULL CHECK (age BETWEEN 0 AND 110),
    age_reference_date DATE NOT NULL,
    CONSTRAINT chk_client_birth_before_reference CHECK (birth_date <= age_reference_date),
    CONSTRAINT chk_client_age_consistent CHECK (
        age = EXTRACT(YEAR FROM AGE(age_reference_date, birth_date))
    ),
    district_id INTEGER NOT NULL,

    CONSTRAINT fk_client_district
        FOREIGN KEY (district_id)
        REFERENCES district(district_id)
);


-- ============================================================
-- DISPOSITION
-- Relationship between clients and accounts.
-- OWNER     = account owner
-- DISPONENT = authorized user
-- ============================================================

CREATE TABLE disp (
    disp_id INTEGER PRIMARY KEY,
    client_id INTEGER NOT NULL,
    account_id INTEGER NOT NULL,
    type TEXT NOT NULL,

    CONSTRAINT fk_disp_client
        FOREIGN KEY (client_id)
        REFERENCES client(client_id),

    CONSTRAINT fk_disp_account
        FOREIGN KEY (account_id)
        REFERENCES account(account_id),

    CONSTRAINT chk_disp_type
        CHECK (
            type IN (
                'OWNER',
                'DISPONENT'
            )
        )
);


-- ============================================================
-- CARD
-- Credit cards associated with dispositions.
-- ============================================================

CREATE TABLE card (
    card_id INTEGER PRIMARY KEY,
    disp_id INTEGER NOT NULL,
    type TEXT NOT NULL,
    issued DATE NOT NULL,

    CONSTRAINT fk_card_disp
        FOREIGN KEY (disp_id)
        REFERENCES disp(disp_id),

    CONSTRAINT chk_card_type
        CHECK (
            type IN (
                'junior',
                'classic',
                'gold'
            )
        )
);


-- ============================================================
-- STANDING ORDER
-- Permanent payment orders associated with bank accounts.
-- k_symbol values are standardized during the ETL process.
-- ============================================================

CREATE TABLE standing_order (
    order_id INTEGER PRIMARY KEY,
    account_id INTEGER NOT NULL,
    bank_to TEXT NOT NULL,
    account_to BIGINT NOT NULL,
    amount NUMERIC NOT NULL,
    k_symbol TEXT,

    CONSTRAINT fk_order_account
        FOREIGN KEY (account_id)
        REFERENCES account(account_id),

    CONSTRAINT chk_standing_order_k_symbol
        CHECK (
            k_symbol IN (
                'INSURANCE_PAYMENT',
                'HOUSEHOLD_PAYMENT',
                'LEASING_PAYMENT',
                'LOAN_PAYMENT'
            )
        )
);


-- ============================================================
-- LOAN
-- Loans associated with bank accounts.
-- Each account can have at most one loan in the Berka model.
-- Original loan status codes are preserved:
-- A, B, C, D.
-- ============================================================

CREATE TABLE loan (
    loan_id INTEGER PRIMARY KEY,
    account_id INTEGER NOT NULL UNIQUE,
    date DATE NOT NULL,
    amount NUMERIC NOT NULL,
    duration INTEGER NOT NULL,
    payments NUMERIC NOT NULL,
    status TEXT NOT NULL,

    CONSTRAINT fk_loan_account
        FOREIGN KEY (account_id)
        REFERENCES account(account_id),

    CONSTRAINT chk_loan_status
        CHECK (
            status IN (
                'A',
                'B',
                'C',
                'D'
            )
        )
);


-- ============================================================
-- BANK TRANSACTION
-- Historical account transactions.
-- Categorical values are standardized during the ETL process.
-- ============================================================

CREATE TABLE bank_transaction (
    trans_id INTEGER PRIMARY KEY,
    account_id INTEGER NOT NULL,
    date DATE NOT NULL,
    type TEXT NOT NULL,
    operation TEXT,
    amount NUMERIC NOT NULL,
    balance NUMERIC NOT NULL,
    k_symbol TEXT,
    counterparty_bank TEXT,
    counterparty_account BIGINT,

    CONSTRAINT fk_transaction_account
        FOREIGN KEY (account_id)
        REFERENCES account(account_id),

    CONSTRAINT chk_transaction_type
        CHECK (
            type IN (
                'CREDIT',
                'DEBIT',
                'WITHDRAWAL'
            )
        ),

    CONSTRAINT chk_transaction_operation
        CHECK (
            operation IN (
                'CASH_DEPOSIT',
                'CASH_WITHDRAWAL',
                'CARD_WITHDRAWAL',
                'TRANSFER_IN',
                'TRANSFER_OUT'
            )
        ),

    CONSTRAINT chk_transaction_k_symbol
        CHECK (
            k_symbol IN (
                'INSURANCE_PAYMENT',
                'STATEMENT_PAYMENT',
                'INTEREST_CREDITED',
                'PENALTY_INTEREST',
                'HOUSEHOLD_PAYMENT',
                'PENSION',
                'LOAN_PAYMENT'
            )
        )
);

COMMENT ON COLUMN client.age IS 'Completed years at age_reference_date; descriptive historical snapshot, not age at origination.';
COMMENT ON COLUMN client.age_reference_date IS 'Maximum transaction date in the loaded source extract; recomputed by ETL.';
COMMENT ON COLUMN client.gender IS 'Sex encoded in Berka birth_number; Female for months 51–62, Male for 01–12.';
COMMENT ON COLUMN client.birth_date IS 'Decoded Berka date; century explicitly assumed to be 1900 for this historical extract.';
