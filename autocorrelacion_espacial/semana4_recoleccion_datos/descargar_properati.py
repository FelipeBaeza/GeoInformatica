#!/usr/bin/env python3
"""
Descarga datos reales de Properati (API pública)
Dataset: Propiedades en Chile con precios reales
"""

import requests
import pandas as pd
from datetime import datetime
import os

def descargar_properati():
    """Descarga dataset de Properati Chile"""
    
    print("=" * 80)
    print("📥 DESCARGANDO DATOS REALES DE PROPERATI")
    print("=" * 80)
    
    # URL del dataset público de Chile
    url = "https://s3.amazonaws.com/properati-data-public/cl_properties.csv.gz"
    
    try:
        print(f"\n🌐 Conectando a: {url}")
        print("⏳ Descargando... (puede tomar 1-2 minutos)")
        
        # Descargar con compression automática
        df = pd.read_csv(url, compression='gzip', low_memory=False)
        
        print(f"\n✅ Descargado exitosamente!")
        print(f"📊 Total propiedades: {len(df):,}")
        print(f"📋 Columnas: {len(df.columns)}")
        
        # Info temporal
        if 'created_on' in df.columns:
            df['created_on'] = pd.to_datetime(df['created_on'])
            print(f"🗓️  Periodo: {df['created_on'].min().date()} a {df['created_on'].max().date()}")
        
        # Filtrar Región Metropolitana
        print("\n🔍 Filtrando Región Metropolitana...")
        
        if 'l2' in df.columns:
            df_rm = df[df['l2'].str.contains('Región Metropolitana', na=False, case=False)]
        elif 'state_name' in df.columns:
            df_rm = df[df['state_name'].str.contains('Metropolitana', na=False, case=False)]
        else:
            print("⚠️  No se encontró columna de región, usando todos los datos")
            df_rm = df
        
        print(f"📍 Propiedades en RM: {len(df_rm):,}")
        
        # Filtrar por comunas de interés
        comunas_interes = ['La Reina', 'Ñuñoa', 'Santiago', 'Estación Central']
        
        if 'l3' in df_rm.columns:
            df_filtrado = df_rm[df_rm['l3'].str.contains('|'.join(comunas_interes), na=False, case=False)]
            print(f"🎯 Propiedades en comunas objetivo: {len(df_filtrado):,}")
        else:
            df_filtrado = df_rm
        
        # Crear directorio de salida
        output_dir = 'datos_procesados'
        os.makedirs(output_dir, exist_ok=True)
        
        # Guardar dataset completo RM
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file_rm = f'{output_dir}/properati_rm_{timestamp}.csv'
        df_rm.to_csv(output_file_rm, index=False)
        print(f"\n💾 Dataset RM guardado: {output_file_rm}")
        
        # Guardar dataset filtrado si existe
        if len(df_filtrado) > 0:
            output_file_filtrado = f'{output_dir}/properati_comunas_{timestamp}.csv'
            df_filtrado.to_csv(output_file_filtrado, index=False)
            print(f"💾 Dataset comunas guardado: {output_file_filtrado}")
        
        # Mostrar estadísticas
        print("\n" + "=" * 80)
        print("📊 ESTADÍSTICAS DEL DATASET")
        print("=" * 80)
        
        print(f"\n📋 Columnas disponibles ({len(df_rm.columns)}):")
        for col in df_rm.columns:
            print(f"   • {col}")
        
        # Estadísticas de precios
        if 'price' in df_rm.columns:
            precios = df_rm['price'].dropna()
            if len(precios) > 0:
                print(f"\n💰 Precios (moneda original):")
                print(f"   • Mínimo: {precios.min():,.0f}")
                print(f"   • Promedio: {precios.mean():,.0f}")
                print(f"   • Mediana: {precios.median():,.0f}")
                print(f"   • Máximo: {precios.max():,.0f}")
        
        # Tipos de propiedad
        if 'property_type' in df_rm.columns:
            print(f"\n🏠 Tipos de propiedad:")
            tipos = df_rm['property_type'].value_counts().head(5)
            for tipo, count in tipos.items():
                print(f"   • {tipo}: {count:,} ({count/len(df_rm)*100:.1f}%)")
        
        # Operaciones
        if 'operation_type' in df_rm.columns:
            print(f"\n📝 Tipos de operación:")
            ops = df_rm['operation_type'].value_counts()
            for op, count in ops.items():
                print(f"   • {op}: {count:,} ({count/len(df_rm)*100:.1f}%)")
        
        print("\n" + "=" * 80)
        print("✅ DESCARGA COMPLETADA")
        print("=" * 80)
        
        return df_rm
        
    except Exception as e:
        print(f"\n❌ Error al descargar: {e}")
        print("\n💡 ALTERNATIVAS:")
        print("\n1️⃣  Descarga manual desde:")
        print("   https://www.properati.com.ar/data/")
        print("   Busca: 'cl_properties.csv.gz' (Chile)")
        
        print("\n2️⃣  Usa datos de años anteriores (si el link cambió)")
        
        print("\n3️⃣  APIs alternativas:")
        print("   • Mercado Libre API")
        print("   • Toctoc (si tienen API)")
        
        return None

if __name__ == "__main__":
    df = descargar_properati()
    
    if df is not None:
        print(f"\n🎉 Dataset listo para usar con {len(df):,} propiedades reales!")
        print("\n📍 Siguiente paso: geocodificar_propiedades.py")
