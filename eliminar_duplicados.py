"""
Script para eliminar filas duplicadas de archivos CSV de propiedades.
Elimina duplicados donde TODAS las columnas son exactamente iguales.
Genera archivo con formato: Ubicaciones_[Comuna].csv
"""

import csv
import os
from datetime import datetime


def obtener_comunas_del_csv(archivo):
    """Lee el CSV y obtiene las comunas únicas presentes en los datos."""
    comunas = set()
    with open(archivo, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f, delimiter=';')
        for fila in reader:
            comuna = fila.get('comuna', '').strip()
            if comuna:
                comunas.add(comuna)
    return sorted(comunas)


def obtener_tipos_propiedad_del_csv(archivo):
    """Lee el CSV y obtiene los tipos de propiedad únicos presentes en los datos."""
    tipos = set()
    with open(archivo, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f, delimiter=';')
        for fila in reader:
            tipo = fila.get('tipo_propiedad', '').strip()
            if tipo:
                tipos.add(tipo)
    return sorted(tipos)


def generar_nombre_salida(comunas, tipos):
    """
    Genera el nombre del archivo de salida basado en las comunas y tipos.
    Formato: Ubicaciones_[tipo]_[comuna].csv
    """
    if not comunas:
        comunas = ["sin_comuna"]
    
    if not tipos:
        tipos = ["propiedad"]
    
    # Limpiar nombres de comunas para el archivo (sin espacios ni caracteres especiales)
    comunas_limpias = []
    for comuna in comunas:
        comuna_limpia = comuna.replace(" ", "_").replace("ñ", "n").replace("Ñ", "N")
        comuna_limpia = ''.join(c for c in comuna_limpia if c.isalnum() or c == '_')
        comunas_limpias.append(comuna_limpia)
    
    # Limpiar nombres de tipos
    tipos_limpios = []
    for tipo in tipos:
        tipo_limpio = tipo.replace(" ", "_").replace("ñ", "n").replace("Ñ", "N")
        tipo_limpio = ''.join(c for c in tipo_limpio if c.isalnum() or c == '_')
        # Capitalizar primera letra
        tipo_limpio = tipo_limpio.capitalize() if tipo_limpio else tipo_limpio
        tipos_limpios.append(tipo_limpio)
    
    # Construir nombre de tipos
    if len(tipos_limpios) > 2:
        nombre_tipos = "_".join(tipos_limpios[:2]) + "_y_mas"
    else:
        nombre_tipos = "_".join(tipos_limpios)
    
    # Construir nombre de comunas
    if len(comunas_limpias) > 2:
        nombre_comunas = "_".join(comunas_limpias[:2]) + "_y_mas"
    else:
        nombre_comunas = "_".join(comunas_limpias)
    
    return f"Ubicaciones_{nombre_tipos}_{nombre_comunas}.csv"


def eliminar_duplicados_csv(archivo_entrada):
    """
    Elimina filas duplicadas de un archivo CSV.
    Solo elimina si TODAS las columnas son exactamente iguales.
    Genera archivo con nombre: Ubicaciones_[Tipo]_[Comuna].csv
    """
    # Obtener comunas y tipos para generar nombre descriptivo
    comunas = obtener_comunas_del_csv(archivo_entrada)
    tipos = obtener_tipos_propiedad_del_csv(archivo_entrada)
    archivo_salida = generar_nombre_salida(comunas, tipos)
    
    print("=" * 60)
    print("🧹 ELIMINADOR DE DUPLICADOS")
    print("=" * 60)
    print(f"\n📂 Archivo entrada: {archivo_entrada}")
    print(f"🏘️  Comunas encontradas: {', '.join(comunas)}")
    print(f"🏠 Tipos de propiedad: {', '.join(tipos)}")
    print(f"📂 Archivo salida: {archivo_salida}")
    
    # Leer todas las filas
    with open(archivo_entrada, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f, delimiter=';')
        headers = next(reader)
        filas = list(reader)
    
    total_original = len(filas)
    print(f"📊 Filas originales: {total_original}")
    
    # Eliminar duplicados usando un set de tuplas
    # Convertimos cada fila a tupla para poder usar set
    filas_unicas = []
    filas_vistas = set()
    duplicados_encontrados = 0
    
    for fila in filas:
        fila_tupla = tuple(fila)
        
        if fila_tupla not in filas_vistas:
            filas_vistas.add(fila_tupla)
            filas_unicas.append(fila)
        else:
            duplicados_encontrados += 1
    
    total_final = len(filas_unicas)
    
    print(f"🔍 Duplicados encontrados: {duplicados_encontrados}")
    print(f"✅ Filas únicas: {total_final}")
    
    # Guardar archivo sin duplicados
    with open(archivo_salida, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(headers)
        writer.writerows(filas_unicas)
    
    print(f"\n📁 Archivo guardado: {archivo_salida}")
    print(f"📍 Ubicación: {os.path.abspath(archivo_salida)}")
    
    if duplicados_encontrados > 0:
        print(f"\n💡 Se eliminaron {duplicados_encontrados} duplicados ({duplicados_encontrados/total_original*100:.1f}%)")
    else:
        print("\n✨ No se encontraron duplicados. El archivo está limpio.")
    
    return archivo_salida, duplicados_encontrados


# --- Menú Principal ---
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🧹 ELIMINADOR DE DUPLICADOS CSV")
    print("=" * 60)
    
    # Listar archivos CSV disponibles (propiedades y ubicaciones)
    archivos_csv = [f for f in os.listdir('.') if f.endswith('.csv') and (f.startswith('propiedades_') or f.startswith('Ubicaciones_'))]
    
    if not archivos_csv:
        print("\n❌ No se encontraron archivos CSV.")
    else:
        # Ordenar por fecha de modificación
        archivos_csv.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        
        print("\n📂 Archivos CSV disponibles:\n")
        for i, archivo in enumerate(archivos_csv, 1):
            # Contar filas
            try:
                with open(archivo, 'r', encoding='utf-8-sig') as f:
                    total = sum(1 for _ in f) - 1
            except:
                total = "?"
            print(f"   [{i}] {archivo} ({total} filas)")
        
        print(f"\n   [0] Salir")
        
        try:
            opcion = input("\n👉 Seleccione el archivo a limpiar: ").strip()
            
            if opcion == '0':
                print("\n👋 ¡Hasta luego!")
            else:
                idx = int(opcion) - 1
                if 0 <= idx < len(archivos_csv):
                    archivo = archivos_csv[idx]
                    
                    archivo_salida, duplicados = eliminar_duplicados_csv(archivo)
                    
                    print(f"\n✅ Proceso completado. Archivo creado: {archivo_salida}")
                else:
                    print("\n❌ Opción no válida.")
                    
        except ValueError:
            print("\n❌ Entrada inválida.")
        except KeyboardInterrupt:
            print("\n\n⚠️  Operación cancelada.")
