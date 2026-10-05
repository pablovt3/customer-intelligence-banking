"""Tests of temporal boundaries, outcome eligibility and train-only preprocessing."""
import unittest
import numpy as np
import pandas as pd
from src.credit_risk_model import FEATURES, feature_matrix, make_model, temporal_split, validate_dataset


def sample():
    n = 24
    dates = pd.date_range('1994-01-01', periods=n, freq='30D')
    frame = pd.DataFrame({name: np.arange(n, dtype=float) + 1 for name in FEATURES})
    frame['preloan_negative_balance_ratio'] = 0.1
    frame['loan_id'] = np.arange(n)
    frame['account_id'] = np.arange(n)
    frame['loan_date'] = dates
    frame['account_open_date'] = dates - pd.Timedelta(days=400)
    frame['preloan_first_transaction_date'] = dates - pd.Timedelta(days=300)
    frame['preloan_last_transaction_date'] = dates - pd.Timedelta(days=1)
    frame['loan_status'] = ['A', 'B'] * 12
    frame['target_default'] = pd.array([0, 1] * 12, dtype='Int64')
    return frame


class BoundaryTests(unittest.TestCase):
    def test_active_loans_excluded_not_imputed(self):
        frame = sample()
        frame.loc[0:1, 'loan_status'] = ['C', 'D']
        frame.loc[0:1, 'target_default'] = pd.NA
        self.assertEqual(len(validate_dataset(frame)), 22)
        frame.loc[0, 'target_default'] = 0
        with self.assertRaisesRegex(ValueError, 'Only A/B'):
            validate_dataset(frame)

    def test_rejects_incorrect_outcome(self):
        frame = sample()
        frame.loc[0, 'target_default'] = 1
        with self.assertRaisesRegex(ValueError, 'Target/status mismatch'):
            validate_dataset(frame)

    def test_rejects_same_day_history(self):
        frame = sample()
        frame.loc[0, 'preloan_last_transaction_date'] = frame.loc[0, 'loan_date']
        with self.assertRaisesRegex(ValueError, 'same-day'):
            validate_dataset(frame)

    def test_rejects_duplicate_loans(self):
        frame = sample()
        frame.loc[0, 'loan_id'] = 1
        with self.assertRaisesRegex(ValueError, 'unique'):
            validate_dataset(frame)

    def test_split_keeps_same_dates_together(self):
        frame = sample()
        frame.loc[17, 'loan_date'] = frame.loc[18, 'loan_date']
        train, test = temporal_split(validate_dataset(frame))
        self.assertLess(train.loan_date.max(), test.loan_date.min())
        self.assertEqual(len(train), 17)

    def test_rejects_shared_account(self):
        frame = sample()
        frame.loc[23, 'account_id'] = 0
        with self.assertRaisesRegex(ValueError, 'Accounts overlap'):
            temporal_split(validate_dataset(frame))

    def test_outcomes_and_identifiers_not_predictors(self):
        frame = sample()
        before = feature_matrix(frame)
        frame['loan_status'] = 'B'
        frame['target_default'] = 1
        frame['loan_id'] = 999
        pd.testing.assert_frame_equal(before, feature_matrix(frame))

    def test_preprocessing_not_refit_on_holdout(self):
        train, test = temporal_split(validate_dataset(sample()))
        model = make_model().fit(feature_matrix(train), train.target_default.astype(int))
        means = model.named_steps['scale'].mean_.copy()
        medians = model.named_steps['impute'].statistics_.copy()
        test.loc[:, 'preloan_avg_balance'] = 1e9
        model.predict_proba(feature_matrix(test))
        np.testing.assert_array_equal(means, model.named_steps['scale'].mean_)
        np.testing.assert_array_equal(medians, model.named_steps['impute'].statistics_)

    def test_rejects_infinite_predictor(self):
        frame = sample()
        frame.loc[0, 'preloan_monthly_inflow'] = np.inf
        with self.assertRaisesRegex(ValueError, 'Infinite'):
            feature_matrix(frame)


if __name__ == '__main__':
    unittest.main()
