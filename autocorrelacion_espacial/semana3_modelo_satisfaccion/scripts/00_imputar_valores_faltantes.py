#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 00: Imputación Inteligente de Valores Faltantes
======================================================

Imputa valores faltantes en el dataset de propiedades usando estrategias
contextuales basadas en la lógica del dominio inmobiliario.

Estrategias de imputación:
1. estacionamientos: Predicción basada en superficie, dormitorios, piso
2. m2_por_habitante: Cálculo derivado de superficie_util / cant_max_habitantes
3. razon_banos_dormitorios: Cálculo directo banos/dormitorios

Author: Sistema de Análisis Geoespacial
Date: 2024
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

class ImputadorPropiedades:
    """Clase para imputar valores faltantes en datos de propiedades."""
    
    def __init__(self, ruta_datos):
        """
        Inicializa el imputador.
        
        Parameters
        ----------
        ruta_datos : str o Path
            Ruta al archivo CSV con datos de propiedades
        """
        self.ruta_datos = Path(ruta_datos)
        self.df = None
        self.df_original = None
        self.reporte = {}
        
    def cargar_datos(self):
        """Carga el dataset de propiedades."""
        print("\n📂 Cargando datos...")
        self.df = pd.read_csv(self.ruta_datos)
        self.df_original = self.df.copy()
        print(f"   ✓ Cargadas {len(self.df):,} propiedades × {len(self.df.columns)} columnas")
        
        return self
    
    def analizar_valores_faltantes(self):
        """Genera reporte de valores faltantes antes de imputación."""
        print("\n📊 Analizando valores faltantes...")
        
        features_clave = [
            'estacionamientos', 'm2_por_habitante', 'razon_banos_dormitorios',
            'cant_max_habitantes', 'superficie_util', 'dormitorios', 'banos'
        ]
        
        for feat in features_clave:
            if feat in self.df.columns:
                total = len(self.df)
                nulos = self.df[feat].isnull().sum()
                pct = (nulos/total)*100
                self.reporte[f"{feat}_antes"] = {
                    'nulos': nulos, 
                    'pct': pct
                }
                if pct > 0:
                    print(f"   • {feat}: {nulos:,} nulos ({pct:.1f}%)")
        
        return self
    
    def imputar_cant_max_habitantes(self):
        """
        Imputa cant_max_habitantes basado en dormitorios.
        
        Lógica:
        - 0 dormitorios (studio) → 1 persona
        - 1 dormitorio → 2 personas
        - 2+ dormitorios → dormitorios + 1
        """
        print("\n🏠 Imputando cant_max_habitantes...")
        
        if 'cant_max_habitantes' not in self.df.columns:
            print("   ⚠ Columna 'cant_max_habitantes' no existe, se creará")
            self.df['cant_max_habitantes'] = np.nan
        
        # Contar nulos Y ceros (ambos necesitan imputación)
        mask_invalidos = self.df['cant_max_habitantes'].isnull() | (self.df['cant_max_habitantes'] == 0)
        invalidos_antes = mask_invalidos.sum()
        
        # Estrategia: estimar por dormitorios
        def calcular_habitantes(row):
            # Si ya tiene un valor válido (>0), mantenerlo
            if pd.notna(row['cant_max_habitantes']) and row['cant_max_habitantes'] > 0:
                return row['cant_max_habitantes']
            
            # Calcular basado en dormitorios
            dorm = row.get('dormitorios', np.nan)
            if pd.isna(dorm):
                return 2  # Valor por defecto conservador
            
            dorm = int(dorm)
            if dorm == 0:
                return 1  # Studio
            elif dorm == 1:
                return 2
            else:
                return dorm + 1
        
        self.df['cant_max_habitantes'] = self.df.apply(calcular_habitantes, axis=1)
        
        invalidos_despues = (self.df['cant_max_habitantes'].isnull() | (self.df['cant_max_habitantes'] == 0)).sum()
        imputados = invalidos_antes - invalidos_despues
        
        print(f"   ✓ Imputados {imputados:,} valores (nulos + ceros)")
        print(f"   📊 Distribución: μ={self.df['cant_max_habitantes'].mean():.1f}, "
              f"σ={self.df['cant_max_habitantes'].std():.1f}")
        
        return self
    
    def imputar_m2_por_habitante(self):
        """
        Imputa m2_por_habitante calculándolo directamente.
        
        Lógica: m2_por_habitante = superficie_util / cant_max_habitantes
        """
        print("\n📐 Imputando m2_por_habitante...")
        
        if 'm2_por_habitante' not in self.df.columns:
            self.df['m2_por_habitante'] = np.nan
        
        mask_nulos = self.df['m2_por_habitante'].isnull()
        nulos_antes = mask_nulos.sum()
        
        # Calcular donde falta
        mask_puede_calcular = (
            mask_nulos & 
            self.df['superficie_util'].notna() & 
            self.df['cant_max_habitantes'].notna() &
            (self.df['cant_max_habitantes'] > 0)
        )
        
        self.df.loc[mask_puede_calcular, 'm2_por_habitante'] = (
            self.df.loc[mask_puede_calcular, 'superficie_util'] / 
            self.df.loc[mask_puede_calcular, 'cant_max_habitantes']
        )
        
        # Para valores extremos, aplicar límites razonables
        # OMS recomienda mínimo 9-12 m²/persona
        self.df.loc[self.df['m2_por_habitante'] < 5, 'm2_por_habitante'] = 5.0
        self.df.loc[self.df['m2_por_habitante'] > 200, 'm2_por_habitante'] = 200.0
        
        nulos_despues = self.df['m2_por_habitante'].isnull().sum()
        imputados = nulos_antes - nulos_despues
        
        print(f"   ✓ Imputados {imputados:,} valores")
        print(f"   📊 Distribución: μ={self.df['m2_por_habitante'].mean():.1f} m²/persona, "
              f"σ={self.df['m2_por_habitante'].std():.1f}")
        
        return self
    
    def imputar_razon_banos_dormitorios(self):
        """
        Imputa razon_banos_dormitorios calculándolo directamente.
        
        Lógica: razon = banos / max(dormitorios, 1)
        """
        print("\n🚿 Imputando razon_banos_dormitorios...")
        
        if 'razon_banos_dormitorios' not in self.df.columns:
            self.df['razon_banos_dormitorios'] = np.nan
        
        mask_nulos = self.df['razon_banos_dormitorios'].isnull()
        nulos_antes = mask_nulos.sum()
        
        # Calcular donde falta
        mask_puede_calcular = (
            mask_nulos & 
            self.df['banos'].notna() & 
            self.df['dormitorios'].notna()
        )
        
        # Evitar división por cero usando max(dormitorios, 1)
        self.df.loc[mask_puede_calcular, 'razon_banos_dormitorios'] = (
            self.df.loc[mask_puede_calcular, 'banos'] / 
            self.df.loc[mask_puede_calcular, 'dormitorios'].clip(lower=1)
        )
        
        nulos_despues = self.df['razon_banos_dormitorios'].isnull().sum()
        imputados = nulos_antes - nulos_despues
        
        print(f"   ✓ Imputados {imputados:,} valores")
        print(f"   📊 Distribución: μ={self.df['razon_banos_dormitorios'].mean():.2f}, "
              f"σ={self.df['razon_banos_dormitorios'].std():.2f}")
        
        return self
    
    def imputar_estacionamientos(self):
        """
        Imputa estacionamientos usando Random Forest.
        
        Predictores: superficie_util, dormitorios, banos, numero_piso_unidad,
                     tiene_estacionamiento
        """
        print("\n🚗 Imputando estacionamientos...")
        
        if 'estacionamientos' not in self.df.columns:
            print("   ⚠ Columna 'estacionamientos' no existe")
            return self
        
        # Features predictivos
        predictores = [
            'superficie_util', 'dormitorios', 'banos', 
            'numero_piso_unidad', 'tiene_estacionamiento'
        ]
        
        # Verificar que existan
        predictores_disponibles = [p for p in predictores if p in self.df.columns]
        
        if len(predictores_disponibles) < 3:
            print(f"   ⚠ Insuficientes predictores ({len(predictores_disponibles)}), usando mediana")
            mediana = self.df['estacionamientos'].median()
            self.df['estacionamientos'].fillna(mediana, inplace=True)
            return self
        
        # Separar datos de entrenamiento (con estacionamientos) y a imputar (sin)
        mask_train = self.df['estacionamientos'].notna()
        mask_predict = self.df['estacionamientos'].isnull()
        
        # Verificar que haya datos válidos en predictores
        for pred in predictores_disponibles:
            if self.df[pred].isnull().any():
                # Imputar predictores faltantes con mediana/moda
                if self.df[pred].dtype in ['float64', 'int64']:
                    self.df[pred].fillna(self.df[pred].median(), inplace=True)
                else:
                    self.df[pred].fillna(self.df[pred].mode()[0], inplace=True)
        
        X_train = self.df.loc[mask_train, predictores_disponibles]
        y_train = self.df.loc[mask_train, 'estacionamientos']
        X_predict = self.df.loc[mask_predict, predictores_disponibles]
        
        nulos_antes = mask_predict.sum()
        
        # Verificar si hay valores a imputar
        if nulos_antes == 0:
            print(f"   ✓ No hay valores faltantes en estacionamientos")
            return self
        
        if len(X_train) < 10:
            print(f"   ⚠ Pocos datos de entrenamiento ({len(X_train)}), usando mediana")
            mediana = y_train.median()
            self.df.loc[mask_predict, 'estacionamientos'] = mediana
        else:
            # Entrenar Random Forest
            rf = RandomForestRegressor(
                n_estimators=100, 
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
            
            rf.fit(X_train, y_train)
            
            # Predecir
            predicciones = rf.predict(X_predict)
            
            # Redondear a enteros y aplicar límites lógicos
            predicciones = np.round(predicciones).astype(int)
            predicciones = np.clip(predicciones, 0, 6)  # Máximo 6 estacionamientos
            
            self.df.loc[mask_predict, 'estacionamientos'] = predicciones
            
            print(f"   ✓ Modelo RF entrenado con {len(X_train):,} propiedades (R²={rf.score(X_train, y_train):.3f})")
        
        imputados = nulos_antes
        print(f"   ✓ Imputados {imputados:,} valores")
        print(f"   📊 Distribución: μ={self.df['estacionamientos'].mean():.1f}, "
              f"σ={self.df['estacionamientos'].std():.1f}")
        
        return self
    
    def validar_imputacion(self):
        """Valida calidad de la imputación."""
        print("\n✅ Validando imputación...")
        
        features_clave = [
            'superficie_total', 'superficie_util', 'dormitorios', 'banos',
            'estacionamientos', 'ambientes', 'numero_piso_unidad',
            'tiene_estacionamiento', 'tiene_bodega', 'm2_por_habitante',
            'razon_banos_dormitorios'
        ]
        
        # Contar propiedades completas
        df_completo_antes = self.df_original[features_clave].dropna()
        df_completo_despues = self.df[features_clave].dropna()
        
        print(f"\n   Propiedades completas:")
        print(f"   • Antes: {len(df_completo_antes):,} ({len(df_completo_antes)/len(self.df)*100:.1f}%)")
        print(f"   • Después: {len(df_completo_despues):,} ({len(df_completo_despues)/len(self.df)*100:.1f}%)")
        print(f"   • Ganancia: +{len(df_completo_despues) - len(df_completo_antes):,} propiedades "
              f"({(len(df_completo_despues) - len(df_completo_antes))/len(self.df)*100:.1f}%)")
        
        # Verificar valores nulos restantes
        nulos_totales = self.df[features_clave].isnull().sum().sum()
        print(f"\n   Valores nulos restantes: {nulos_totales:,}")
        
        if nulos_totales > 0:
            print("\n   ⚠ Factores con nulos restantes:")
            for feat in features_clave:
                if feat in self.df.columns:
                    nulos = self.df[feat].isnull().sum()
                    if nulos > 0:
                        print(f"      • {feat}: {nulos:,}")
        
        return self
    
    def guardar_dataset_imputado(self):
        """Guarda dataset con valores imputados."""
        print("\n💾 Guardando dataset imputado...")
        
        # Crear backup del original
        backup_path = self.ruta_datos.parent / f"{self.ruta_datos.stem}_sin_imputar{self.ruta_datos.suffix}"
        if not backup_path.exists():
            self.df_original.to_csv(backup_path, index=False)
            print(f"   ✓ Backup guardado en: {backup_path.name}")
        
        # Guardar imputado
        self.df.to_csv(self.ruta_datos, index=False)
        print(f"   ✓ Dataset imputado guardado en: {self.ruta_datos.name}")
        print(f"   📊 {len(self.df):,} propiedades × {len(self.df.columns)} columnas")
        
        return self
    
    def generar_reporte(self):
        """Genera reporte detallado de imputación."""
        print("\n" + "="*80)
        print("📊 REPORTE DE IMPUTACIÓN")
        print("="*80)
        
        print(f"\n🗂️  Dataset: {self.ruta_datos.name}")
        print(f"📈 Total propiedades: {len(self.df):,}")
        
        print("\n┌─ Mejora en Completitud ─────────────────────────────────────────────┐")
        
        features = [
            'superficie_total', 'superficie_util', 'dormitorios', 'banos',
            'estacionamientos', 'ambientes', 'numero_piso_unidad',
            'tiene_estacionamiento', 'tiene_bodega', 'm2_por_habitante',
            'razon_banos_dormitorios'
        ]
        
        # Antes vs Después
        completas_antes = len(self.df_original[features].dropna())
        completas_despues = len(self.df[features].dropna())
        mejora = completas_despues - completas_antes
        mejora_pct = (mejora / len(self.df)) * 100
        
        print(f"│ Propiedades con TODOS los factores completos:")
        print(f"│   • ANTES:   {completas_antes:>5} ({completas_antes/len(self.df)*100:>5.1f}%)")
        print(f"│   • DESPUÉS: {completas_despues:>5} ({completas_despues/len(self.df)*100:>5.1f}%)")
        print(f"│   • MEJORA:  +{mejora:>4} ({mejora_pct:>5.1f}%) 🎉")
        print("└──────────────────────────────────────────────────────────────────────┘")
        
        print("\n✅ Imputación completada exitosamente!")
        
        return self


def main():
    """Función principal."""
    print("\n" + "="*80)
    print("🔧 IMPUTACIÓN DE VALORES FALTANTES")
    print("="*80)
    
    # Rutas
    BASE_DIR = Path(__file__).parent.parent
    RUTA_DATOS = BASE_DIR / 'data' / 'propiedades_con_factores_espaciales.csv'
    
    # Verificar que existe el archivo
    if not RUTA_DATOS.exists():
        print(f"\n❌ ERROR: No se encuentra el archivo {RUTA_DATOS}")
        print("   Ejecuta primero el script 01_integrar_datos.py")
        return
    
    # Pipeline de imputación
    imputador = ImputadorPropiedades(RUTA_DATOS)
    
    (imputador
     .cargar_datos()
     .analizar_valores_faltantes()
     .imputar_cant_max_habitantes()
     .imputar_m2_por_habitante()
     .imputar_razon_banos_dormitorios()
     .imputar_estacionamientos()
     .validar_imputacion()
     .guardar_dataset_imputado()
     .generar_reporte())
    
    print("\n" + "="*80)
    print("🎯 Siguiente paso: Ejecutar script 02_modelo_satisfaccion.py")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
