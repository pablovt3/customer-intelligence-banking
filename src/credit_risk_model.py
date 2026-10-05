"""Reproducible educational credit-risk baseline and batch scoring.

Run from the project root: python -m src.credit_risk_model train
No database connection is needed: input is the existing analytical parquet.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, precision_score, recall_score,
                             roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
FEATURES = [
    'loan_amount', 'loan_duration', 'account_tenure_days',
    'preloan_avg_balance', 'preloan_min_balance', 'preloan_balance_std',
    'preloan_negative_balance_ratio', 'preloan_transactions_per_month',
    'preloan_monthly_inflow', 'preloan_monthly_outflow',
]
THRESHOLD = 0.5  # Fixed before examining the holdout; not a lending decision rule.


def require(condition, message):
    if not condition:
        raise ValueError(message)


def feature_matrix(frame):
    missing = sorted(set(FEATURES) - set(frame))
    require(not missing, f'Missing predictor columns: {missing}')
    x = frame[FEATURES].apply(pd.to_numeric, errors='raise').astype(float)
    require(not np.isinf(x.to_numpy()).any(), 'Infinite predictors are not allowed.')
    require(x['loan_amount'].gt(0).all(), 'Loan amount must be positive.')
    require(x['loan_duration'].gt(0).all(), 'Loan duration must be positive.')
    require(x['account_tenure_days'].ge(0).all(), 'Account tenure must be non-negative.')
    require(x['preloan_negative_balance_ratio'].dropna().between(0, 1).all(),
            'Negative-balance ratio must lie in [0, 1].')
    return x


def validate_dataset(frame):
    required = {'loan_id', 'account_id', 'loan_date', 'loan_status', 'target_default',
                'account_open_date', 'preloan_first_transaction_date',
                'preloan_last_transaction_date'}
    require(required <= set(frame), f'Missing audit columns: {sorted(required - set(frame))}')
    require(frame.loan_id.notna().all() and frame.loan_id.is_unique, 'Loan IDs must be unique and non-null.')
    require(frame.account_id.notna().all(), 'Account IDs cannot be missing.')
    require(frame.loan_date.notna().all(), 'Loan dates cannot be missing.')
    require(frame.loan_status.isin(['A', 'B', 'C', 'D']).all(), 'Unknown loan status.')
    closed = frame.loan_status.isin(['A', 'B'])
    require(frame.target_default.notna().equals(closed), 'Only A/B may have definitive targets; C/D must be missing.')
    require(frame.loc[closed, 'target_default'].eq(
        frame.loc[closed, 'loan_status'].map({'A': 0, 'B': 1})).all(), 'Target/status mismatch.')
    first, last = frame.preloan_first_transaction_date, frame.preloan_last_transaction_date
    require(first.notna().equals(last.notna()), 'History endpoints must be present together.')
    has_history = last.notna()
    require(last[has_history].lt(frame.loc[has_history, 'loan_date']).all(),
            'History includes same-day or future transactions.')
    require(first[has_history].ge(frame.loc[has_history, 'account_open_date']).all(),
            'History precedes account opening.')
    require(first[has_history].le(last[has_history]).all(), 'Invalid history interval.')
    feature_matrix(frame)
    return frame.loc[closed].sort_values(['loan_date', 'loan_id']).copy()


def temporal_split(cohort):
    require(len(cohort) >= 8, 'Too few completed loans for a temporal holdout.')
    cutoff = cohort.iloc[int(len(cohort) * 0.75)].loan_date
    train = cohort.loc[cohort.loan_date < cutoff].copy()
    test = cohort.loc[cohort.loan_date >= cutoff].copy()
    require(len(train) > 0 and len(test) > 0, 'Temporal split is empty.')
    require(train.loan_date.max() < test.loan_date.min(), 'Temporal split overlaps.')
    require(not set(train.account_id) & set(test.account_id), 'Accounts overlap across train and holdout.')
    for name, part in [('train', train), ('holdout', test)]:
        require(part.target_default.nunique() == 2, f'{name} needs both outcome classes.')
    return train, test


def make_model():
    return Pipeline([
        ('impute', SimpleImputer(strategy='median', keep_empty_features=True)),
        ('scale', StandardScaler()),
        ('model', LogisticRegression(C=1.0, max_iter=3000, random_state=42)),
    ])


def metrics(y, p):
    pred = p >= THRESHOLD
    return {
        'roc_auc': float(roc_auc_score(y, p)),
        'average_precision': float(average_precision_score(y, p)),
        'brier_score': float(brier_score_loss(y, p)),
        'precision_at_0_5': float(precision_score(y, pred, zero_division=0)),
        'recall_at_0_5': float(recall_score(y, pred, zero_division=0)),
        'confusion_matrix_tn_fp_fn_tp': confusion_matrix(y, pred, labels=[0, 1]).ravel().tolist(),
    }


def train_model(data_path, output_root):
    frame = pd.read_parquet(data_path)
    cohort = validate_dataset(frame)
    train, test = temporal_split(cohort)
    x_train, x_test = feature_matrix(train), feature_matrix(test)
    require(not x_train.isna().all().any(), 'A training predictor is entirely missing.')
    y_train, y_test = train.target_default.astype(int), test.target_default.astype(int)
    model = make_model().fit(x_train, y_train)
    dummy = DummyClassifier(strategy='prior').fit(x_train, y_train)
    predictions = model.predict_proba(x_test)[:, 1]
    baseline_predictions = dummy.predict_proba(x_test)[:, 1]
    def describe(part):
        return {'rows': len(part), 'bad': int(part.target_default.sum()),
                'first_loan_date': str(part.loan_date.min().date()),
                'last_loan_date': str(part.loan_date.max().date())}
    result = {
        'scope': 'Educational retrospective baseline on completed loans; not a calibrated PD.',
        'dataset_sha256': hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
        'all_loans': len(frame), 'excluded_active_loans': int(frame.target_default.isna().sum()),
        'train': describe(train), 'holdout': describe(test), 'features': FEATURES,
        'threshold': THRESHOLD, 'logistic_regression': metrics(y_test, predictions),
        'prior_baseline': metrics(y_test, baseline_predictions),
        'versions': {p: importlib.metadata.version(p) for p in
                     ['pandas', 'numpy', 'scikit-learn', 'joblib', 'pyarrow']},
    }
    report_dir, model_dir = output_root / 'reports', output_root / 'models'
    report_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    artifact = {'pipeline': model, 'features': FEATURES, 'metadata': result}
    model_path = model_dir / 'credit_risk_baseline.joblib'
    joblib.dump(artifact, model_path)
    restored = joblib.load(model_path)
    np.testing.assert_allclose(restored['pipeline'].predict_proba(x_test)[:, 1], predictions)
    (report_dir / 'credit_risk_metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    output = test[['loan_id', 'loan_date', 'target_default']].copy()
    output['risk_score'] = predictions
    output['baseline_score'] = baseline_predictions
    output['flag_at_0_5'] = predictions >= THRESHOLD
    output.to_csv(report_dir / 'holdout_predictions.csv', index=False)
    pd.DataFrame({'feature': FEATURES, 'standardized_coefficient':
                  model.named_steps['model'].coef_[0]}).sort_values(
        'standardized_coefficient', ascending=False).to_csv(
            report_dir / 'model_coefficients.csv', index=False)
    write_report(result, report_dir)
    print(json.dumps(result, indent=2))
    return result


def write_report(result, report_dir):
    rows = []
    for key, label in [('roc_auc', 'ROC AUC ↑'), ('average_precision', 'Average precision ↑'),
                       ('brier_score', 'Brier score ↓'), ('precision_at_0_5', 'Precisión a 0,5'),
                       ('recall_at_0_5', 'Sensibilidad a 0,5')]:
        rows.append(f"| {label} | {result['logistic_regression'][key]:.4f} | {result['prior_baseline'][key]:.4f} |")
    tr, te = result['train'], result['holdout']
    report = f'''# Customer Intelligence Suite — Retail Banking

## Riesgo crediticio — evaluación retrospectiva

**Próximos pasos:** evaluar la estabilidad temporal del modelo y revisar la definición del objetivo antes de ampliar el análisis.

Regresión logística regularizada con {len(FEATURES)} variables de contrato e historial previo al préstamo.
La referencia asigna a todos la frecuencia de impago del conjunto de entrenamiento.
El modelo guardado se entrenó exclusivamente con el bloque de entrenamiento; no se reajustó con la evaluación.

## Población y separación

- Total: {result['all_loans']} préstamos; {result['excluded_active_loans']} activos excluidos del entrenamiento y evaluación.
- Entrenamiento: {tr['rows']} préstamos, {tr['bad']} impagos, {tr['first_loan_date']} a {tr['last_loan_date']}.
- Evaluación: {te['rows']} préstamos, **{te['bad']} impagos**, {te['first_loan_date']} a {te['last_loan_date']}.
- Corte por fecha de originación, aproximadamente 75/25; una misma fecha nunca se divide.
- A=0 y B=1. C/D no tienen desenlace definitivo y no se imputan.

## Resultados fuera de entrenamiento

| Métrica | Regresión logística | Referencia |
|---|---:|---:|
{chr(10).join(rows)}

Matriz [verdaderos negativos, falsos positivos, falsos negativos, verdaderos positivos]:
`{result['logistic_regression']['confusion_matrix_tn_fp_fn_tp']}`.
Umbral fijo de 0,5, sin optimizar con la evaluación y sin convertirlo en política de aprobación.

## Alcance y límites

Esta entrega completa un flujo educativo: datos analíticos → entrenamiento → evaluación → modelo guardado → puntuación por lote.
Con solo {te['bad']} impagos en evaluación, las métricas son inestables; no prueban capacidad de generalización.
La selección de contratos finalizados introduce sesgo de maduración y de duración.
Separar por originación no asegura que los desenlaces de entrenamiento ya fueran conocidos en esa fecha:
no se dispone de fechas exactas de observación del impago. Esto es una evaluación retrospectiva,
no una simulación histórica de despliegue ni una probabilidad de impago a un horizonte fijo.
El score no está calibrado para uso bancario real.

Imputación y escalado se ajustan solo con entrenamiento. Estado del préstamo, objetivo, identificadores,
fechas y variables demográficas quedan fuera de los predictores. Se comprueban fechas de historial,
pero esas comprobaciones no sustituyen una nueva auditoría de las transacciones fuente.
Las variables mensuales heredadas usan el intervalo entre primera y última transacción observada.
Los coeficientes estandarizados describen asociaciones, no efectos causales.

## Artefactos

- `credit_risk_metrics.json`: métricas, población, versiones y huella del dataset.
- `holdout_predictions.csv`: predicciones fuera de entrenamiento y etiquetas reales.
- `model_coefficients.csv`: coeficientes por variable.
- `../models/credit_risk_baseline.joblib`: pipeline entrenado y metadatos.

La segmentación de clientes es la siguiente línea de desarrollo del proyecto y se trabajará por separado del modelo de riesgo.
'''
    (report_dir / 'credit_risk_report.md').write_text(report)


def score_batch(model_path, input_path, output_path):
    # Load only locally generated/trusted joblib files: deserialization executes Python.
    artifact = joblib.load(model_path)
    require(artifact.get('features') == FEATURES, 'Model predictor schema is incompatible.')
    frame = pd.read_parquet(input_path) if input_path.suffix == '.parquet' else pd.read_csv(input_path)
    require('loan_id' in frame and frame.loan_id.notna().all() and frame.loan_id.is_unique,
            'Input needs a unique non-null loan_id.')
    require(len(frame) > 0, 'Input is empty.')
    output = frame[['loan_id']].copy()
    output['risk_score'] = artifact['pipeline'].predict_proba(feature_matrix(frame))[:, 1]
    output['flag_at_0_5'] = output.risk_score >= THRESHOLD
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    print(f'Saved {len(output)} educational scores to {output_path}.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    train = sub.add_parser('train')
    train.add_argument('--data', type=Path, default=ROOT / 'data/processed/credit_risk.parquet')
    train.add_argument('--output-root', type=Path, default=ROOT)
    score = sub.add_parser('score')
    score.add_argument('--model', type=Path, default=ROOT / 'models/credit_risk_baseline.joblib')
    score.add_argument('--input', type=Path, required=True)
    score.add_argument('--output', type=Path, default=ROOT / 'reports/batch_scores.csv')
    args = parser.parse_args()
    if args.command == 'train':
        train_model(args.data, args.output_root)
    else:
        score_batch(args.model, args.input, args.output)


if __name__ == '__main__':
    main()
