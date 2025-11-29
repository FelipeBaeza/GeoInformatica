"""
Script para geocodificar direcciones del CSV de propiedades.
Agrega columnas de latitud y longitud usando Nominatim (OpenStreetMap).
"""

import csv
import time
import os
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

# Configuración
DELAY_ENTRE_PETICIONES = 1.5  # Nominatim requiere mínimo 1 segundo entre peticiones
TIMEOUT = 10  # Segundos de timeout por petición


def limpiar_direccion(direccion_raw):
    """
    Limpia y formatea la dirección para mejorar la geocodificación.
    
    Ejemplos de entrada:
    - "Coronel Godoy 0157, Estación Central, San Alberto Hurtado, Estación Central"
    - "Av. Ecuador 3866, San Alberto Hurtado, Estación Central"
    - "VENDO DEPARTAMENTO, BODEGA Y EST. CERCA DEL METRO ECUADOR, Metro Ecuador, Estación Central"
    """
    if not direccion_raw or direccion_raw == "No disponible":
        return None
    
    # Eliminar textos promocionales comunes
    textos_a_eliminar = [
        "VENDO", "DEPARTAMENTO", "BODEGA", "EST.", "CERCA DEL", "REAL OPORTUNIDAD!!",
        "Dpto En Venta", "FRENTE MUTUAL DE SEGURIDAD", "Metro", "METRO",
        "9.1599e+06 - 9.1602e+06", "9170374"
    ]
    
    direccion = direccion_raw
    for texto in textos_a_eliminar:
        direccion = direccion.replace(texto, "")
    
    # Limpiar espacios múltiples y comas consecutivas
    while "  " in direccion:
        direccion = direccion.replace("  ", " ")
    while ",," in direccion:
        direccion = direccion.replace(",,", ",")
    
    direccion = direccion.strip(" ,")
    
    # Tomar las primeras partes relevantes (calle + número + comuna)
    partes = [p.strip() for p in direccion.split(",") if p.strip()]
    
    if len(partes) >= 2:
        # Buscar la comuna en las partes
        comunas_conocidas = ["Estación Central", "Santiago", "Ñuñoa", "La Reina", "Nunoa"]
        comuna = None
        calle = partes[0]
        
        for parte in partes:
            for c in comunas_conocidas:
                if c.lower() in parte.lower():
                    comuna = c
                    break
        
        if comuna:
            # Formato: "Calle Número, Comuna, Santiago, Chile"
            direccion_limpia = f"{calle}, {comuna}, Santiago, Chile"
        else:
            direccion_limpia = f"{calle}, Santiago, Chile"
    else:
        direccion_limpia = f"{direccion}, Santiago, Chile"
    
    return direccion_limpia


def geocodificar_direccion(geolocator, direccion):
    """
    Geocodifica una dirección y retorna (latitud, longitud).
    Retorna (None, None) si no se encuentra.
    """
    if not direccion:
        return None, None
    
    try:
        location = geolocator.geocode(direccion, timeout=TIMEOUT)
        
        if location:
            return location.latitude, location.longitude
        else:
            # Intentar con dirección más simple (solo calle y Chile)
            partes = direccion.split(",")
            if len(partes) > 2:
                direccion_simple = f"{partes[0]}, Chile"
                location = geolocator.geocode(direccion_simple, timeout=TIMEOUT)
                if location:
                    return location.latitude, location.longitude
            
            return None, None
            
    except GeocoderTimedOut:
        print(f"   ⏱️  Timeout para: {direccion[:50]}...")
        return None, None
    except GeocoderServiceError as e:
        print(f"   ❌ Error de servicio: {e}")
        return None, None
    except Exception as e:
        print(f"   ❌ Error inesperado: {e}")
        return None, None


def obtener_comunas_del_csv(archivo):
    """
    Lee el CSV y obtiene las comunas únicas presentes en los datos.
    """
    comunas = set()
    with open(archivo, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f, delimiter=';')
        for fila in reader:
            comuna = fila.get('comuna', '').strip()
            if comuna:
                comunas.add(comuna)
    return sorted(comunas)


def generar_nombre_salida(comunas):
    """
    Genera el nombre del archivo de salida basado en las comunas.
    Formato: Ubicaciones_[comuna1]_[comuna2].csv
    """
    if not comunas:
        return "Ubicaciones_sin_comuna.csv"
    
    # Limpiar nombres de comunas para el archivo (sin espacios ni caracteres especiales)
    comunas_limpias = []
    for comuna in comunas:
        comuna_limpia = comuna.replace(" ", "_").replace("ñ", "n").replace("Ñ", "N")
        comuna_limpia = ''.join(c for c in comuna_limpia if c.isalnum() or c == '_')
        comunas_limpias.append(comuna_limpia)
    
    # Si hay muchas comunas, limitar a las primeras 3
    if len(comunas_limpias) > 3:
        nombre = "_".join(comunas_limpias[:3]) + "_y_mas"
    else:
        nombre = "_".join(comunas_limpias)
    
    return f"Ubicaciones_{nombre}.csv"


def geocodificar_csv(archivo_entrada, archivo_salida=None):
    """
    Lee un CSV de propiedades y agrega columnas de latitud y longitud.
    """
    # Obtener comunas del archivo para generar nombre descriptivo
    comunas = obtener_comunas_del_csv(archivo_entrada)
    
    if not archivo_salida:
        archivo_salida = generar_nombre_salida(comunas)
    
    # Inicializar geocodificador
    geolocator = Nominatim(user_agent="portal_inmobiliario_scraper_v1")
    
    print("=" * 60)
    print("🌍 GEOCODIFICADOR DE PROPIEDADES")
    print("=" * 60)
    print(f"\n📂 Archivo entrada: {archivo_entrada}")
    print(f"🏘️  Comunas encontradas: {', '.join(comunas)}")
    print(f"📂 Archivo salida: {archivo_salida}")
    print(f"⏱️  Delay entre peticiones: {DELAY_ENTRE_PETICIONES}s")
    
    # Leer CSV original
    with open(archivo_entrada, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f, delimiter=';')
        filas = list(reader)
        headers_originales = reader.fieldnames
    
    total_filas = len(filas)
    print(f"\n📊 Total de propiedades: {total_filas}")
    
    # Agregar nuevas columnas
    nuevos_headers = headers_originales + ['latitud', 'longitud', 'direccion_geocoded']
    
    # Procesar y geocodificar
    geocodificadas = 0
    no_encontradas = 0
    
    print("\n🔄 Iniciando geocodificación...\n")
    
    with open(archivo_salida, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=nuevos_headers, delimiter=';')
        writer.writeheader()
        
        for i, fila in enumerate(filas):
            direccion_original = fila.get('ubicacion', '')
            direccion_limpia = limpiar_direccion(direccion_original)
            
            print(f"[{i+1}/{total_filas}] {direccion_original[:60]}...")
            
            if direccion_limpia:
                print(f"   → Buscando: {direccion_limpia}")
                lat, lng = geocodificar_direccion(geolocator, direccion_limpia)
                
                if lat and lng:
                    print(f"   ✅ Encontrado: ({lat:.6f}, {lng:.6f})")
                    geocodificadas += 1
                else:
                    print(f"   ⚠️  No encontrado")
                    no_encontradas += 1
                
                time.sleep(DELAY_ENTRE_PETICIONES)
            else:
                lat, lng = None, None
                no_encontradas += 1
                print(f"   ⚠️  Dirección no válida")
            
            # Agregar datos a la fila
            fila['latitud'] = lat if lat else ""
            fila['longitud'] = lng if lng else ""
            fila['direccion_geocoded'] = direccion_limpia if direccion_limpia else ""
            
            writer.writerow(fila)
    
    # Resumen
    print("\n" + "=" * 60)
    print("📋 RESUMEN GEOCODIFICACIÓN")
    print("=" * 60)
    print(f"\n✅ Geocodificadas exitosamente: {geocodificadas}")
    print(f"⚠️  No encontradas: {no_encontradas}")
    print(f"📊 Tasa de éxito: {geocodificadas/total_filas*100:.1f}%")
    print(f"\n📁 Archivo guardado: {archivo_salida}")
    
    return archivo_salida


# --- Menú Principal ---
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🌍 GEOCODIFICADOR DE DIRECCIONES")
    print("=" * 60)
    
    # Listar archivos CSV disponibles
    archivos_csv = [f for f in os.listdir('.') if f.startswith('propiedades_') and f.endswith('.csv') and '_geocoded' not in f]
    
    if not archivos_csv:
        print("\n❌ No se encontraron archivos CSV de propiedades.")
        print("   Primero ejecute el scrapper para generar datos.")
    else:
        print("\n📂 Archivos CSV disponibles:\n")
        for i, archivo in enumerate(archivos_csv, 1):
            print(f"   [{i}] {archivo}")
        
        print(f"\n   [0] Salir")
        
        try:
            opcion = input("\n👉 Seleccione el archivo a geocodificar: ").strip()
            
            if opcion == '0':
                print("\n👋 ¡Hasta luego!")
            else:
                idx = int(opcion) - 1
                if 0 <= idx < len(archivos_csv):
                    archivo = archivos_csv[idx]
                    
                    # Contar filas para estimar tiempo
                    with open(archivo, 'r', encoding='utf-8-sig') as f:
                        total = sum(1 for _ in f) - 1  # -1 por header
                    
                    tiempo_estimado = total * DELAY_ENTRE_PETICIONES / 60
                    print(f"\n⏱️  Tiempo estimado: {tiempo_estimado:.1f} minutos")
                    
                    confirmar = input("¿Continuar? (s/n): ").strip().lower()
                    
                    if confirmar == 's':
                        geocodificar_csv(archivo)
                    else:
                        print("\n❌ Operación cancelada.")
                else:
                    print("\n❌ Opción no válida.")
                    
        except ValueError:
            print("\n❌ Entrada inválida.")
        except KeyboardInterrupt:
            print("\n\n⚠️  Operación interrumpida.")
