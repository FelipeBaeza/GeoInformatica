#!/usr/bin/env python3
"""Extraer información del modelo para documentación"""

import pickle
import pandas as pd
import numpy as np

# Cargar modelo
with open('modelo_ols_limpio.pkl', 'rb') as f:
    modelo = pickle.load(f)

print('='*70)
print('MODELO OLS LIMPIO - COEFICIENTES DETALLADOS')
print('='*70)
print(f'\nObservaciones: {modelo.nobs:.0f}')
print(f'R²: {modelo.rsquared:.4f}')
print(f'R² ajustado: {modelo.rsquared_adj:.4f}')
print(f'F-statistic: {modelo.fvalue:.2f} (p={modelo.f_pvalue:.6f})')

print(f'\n{"Variable":<35} {"Coef":>10} {"Std Err":>10} {"t":>8} {"P>|t|":>10} {"Sig":>5}')
print('='*70)

for var, coef, stderr, tval, pval in zip(
    modelo.params.index, 
    modelo.params.values,
    modelo.bse.values,
    modelo.tvalues.values,
    modelo.pvalues.values
):
    sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
    print(f'{var:<35} {coef:>10.4f} {stderr:>10.4f} {tval:>8.2f} {pval:>10.6f} {sig:>5}')

print('\n' + '='*70)
print('MÉTRICAS DE BONDAD DE AJUSTE')
print('='*70)
print(f'AIC: {modelo.aic:.2f}')
print(f'BIC: {modelo.bic:.2f}')
print(f'Log-Likelihood: {modelo.llf:.2f}')

# Calcular RMSE
rmse = np.sqrt(modelo.mse_resid)
print(f'RMSE: {rmse:.4f}')
print(f'MAE: {np.abs(modelo.resid).mean():.4f}')
