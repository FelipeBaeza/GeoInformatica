"""
Script para geocodificar direcciones del CSV de propiedades.
Agrega columnas de latitud y longitud usando múltiples servicios de geocodificación.
Soporta procesamiento paralelo para mayor velocidad.
"""

import csv
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from geopy.geocoders import Nominatim, Photon, ArcGIS
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

# Configuración
DELAY_ENTRE_PETICIONES = 0.3  # Delay reducido para Photon
TIMEOUT = 10  # Segundos de timeout por petición
MAX_WORKERS = 5  # Número de hilos paralelos


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


def geocodificar_item(args):
    """
    Geocodifica un item individual. Usado para procesamiento paralelo.
    """
    idx, fila, total, servicio = args
    
    direccion_original = fila.get('ubicacion', '')
    direccion_limpia = limpiar_direccion(direccion_original)
    
    if not direccion_limpia:
        return idx, fila, None, None, None, False
    
    # Crear geolocalizador para este hilo
    try:
        if servicio == 'photon':
            geolocator = Photon(user_agent="portal_inmobiliario_scraper_v1", timeout=TIMEOUT)
        elif servicio == 'arcgis':
            geolocator = ArcGIS(timeout=TIMEOUT)
        else:
            geolocator = Nominatim(user_agent="portal_inmobiliario_scraper_v1", timeout=TIMEOUT)
        
        lat, lng = geocodificar_direccion(geolocator, direccion_limpia)
        
        time.sleep(DELAY_ENTRE_PETICIONES)
        
        return idx, fila, lat, lng, direccion_limpia, True
    except Exception:
        return idx, fila, None, None, direccion_limpia, True


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


def generar_nombre_salida(archivo_entrada):
    """
    Genera el nombre del archivo de salida basado en el nombre de entrada.
    Formato: [nombre_original]_geolocacion.csv
    """
    nombre_base = os.path.splitext(archivo_entrada)[0]
    return f"{nombre_base}_geolocacion.csv"


def geocodificar_csv(archivo_entrada, archivo_salida=None, usar_paralelo=True, servicio='photon'):
    """
    Lee un CSV de propiedades y agrega columnas de latitud y longitud.
    Solo guarda las propiedades que tienen geolocalización encontrada.
    
    Args:
        archivo_entrada: Ruta al archivo CSV
        archivo_salida: Ruta de salida (opcional)
        usar_paralelo: Si True, usa procesamiento paralelo
        servicio: 'photon' (rápido), 'nominatim' (lento pero preciso), 'arcgis'
    """
    if not archivo_salida:
        archivo_salida = generar_nombre_salida(archivo_entrada)
    
    print("=" * 60)
    print("🌍 GEOCODIFICADOR DE PROPIEDADES")
    print("=" * 60)
    print(f"\n📂 Archivo entrada: {archivo_entrada}")
    print(f"📂 Archivo salida: {archivo_salida}")
    print(f"🔧 Servicio: {servicio.upper()}")
    print(f"⚡ Modo paralelo: {'Sí' if usar_paralelo else 'No'} ({MAX_WORKERS} hilos)")
    print(f"⏱️  Delay entre peticiones: {DELAY_ENTRE_PETICIONES}s")
    print(f"⚠️  Solo se guardarán propiedades con geolocalización encontrada")
    
    # Leer CSV original
    with open(archivo_entrada, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f, delimiter=';')
        filas = list(reader)
        headers_originales = reader.fieldnames
    
    total_filas = len(filas)
    print(f"\n📊 Total de propiedades: {total_filas}")
    
    # Estimar tiempo
    if usar_paralelo:
        tiempo_estimado = (total_filas * DELAY_ENTRE_PETICIONES) / MAX_WORKERS / 60
    else:
        tiempo_estimado = total_filas * DELAY_ENTRE_PETICIONES / 60
    print(f"⏱️  Tiempo estimado: {tiempo_estimado:.1f} minutos")
    
    # Agregar nuevas columnas
    nuevos_headers = headers_originales + ['latitud', 'longitud', 'direccion_geocoded']
    
    # Procesar y geocodificar
    geocodificadas = 0
    no_encontradas = 0
    filas_guardadas = []
    
    print("\n🔄 Iniciando geocodificación...\n")
    
    inicio = time.time()
    
    if usar_paralelo:
        # Procesamiento paralelo
        args_list = [(i, fila, total_filas, servicio) for i, fila in enumerate(filas)]
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(geocodificar_item, args): args[0] for args in args_list}
            
            completados = 0
            for future in as_completed(futures):
                try:
                    idx, fila, lat, lng, direccion_limpia, valido = future.result()
                    completados += 1
                    
                    if valido and lat and lng:
                        fila['latitud'] = lat
                        fila['longitud'] = lng
                        fila['direccion_geocoded'] = direccion_limpia
                        filas_guardadas.append((idx, fila))
                        geocodificadas += 1
                        estado = "✅"
                    else:
                        no_encontradas += 1
                        estado = "❌"
                    
                    # Mostrar progreso cada 10 items
                    if completados % 10 == 0 or completados == total_filas:
                        transcurrido = time.time() - inicio
                        velocidad = completados / transcurrido if transcurrido > 0 else 0
                        restante = (total_filas - completados) / velocidad / 60 if velocidad > 0 else 0
                        print(f"   [{completados}/{total_filas}] ✅ {geocodificadas} | ❌ {no_encontradas} | ⏱️ {restante:.1f} min restantes")
                        
                except Exception as e:
                    no_encontradas += 1
        
        # Ordenar por índice original
        filas_guardadas.sort(key=lambda x: x[0])
        filas_guardadas = [fila for _, fila in filas_guardadas]
    
    else:
        # Procesamiento secuencial (original)
        if servicio == 'photon':
            geolocator = Photon(user_agent="portal_inmobiliario_scraper_v1", timeout=TIMEOUT)
        elif servicio == 'arcgis':
            geolocator = ArcGIS(timeout=TIMEOUT)
        else:
            geolocator = Nominatim(user_agent="portal_inmobiliario_scraper_v1", timeout=TIMEOUT)
        
        for i, fila in enumerate(filas):
            direccion_original = fila.get('ubicacion', '')
            direccion_limpia = limpiar_direccion(direccion_original)
            
            if direccion_limpia:
                lat, lng = geocodificar_direccion(geolocator, direccion_limpia)
                
                if lat and lng:
                    geocodificadas += 1
                    fila['latitud'] = lat
                    fila['longitud'] = lng
                    fila['direccion_geocoded'] = direccion_limpia
                    filas_guardadas.append(fila)
                else:
                    no_encontradas += 1
                
                time.sleep(DELAY_ENTRE_PETICIONES)
            else:
                no_encontradas += 1
            
            if (i + 1) % 10 == 0 or (i + 1) == total_filas:
                print(f"   [{i+1}/{total_filas}] ✅ {geocodificadas} | ❌ {no_encontradas}")
    
    tiempo_total = time.time() - inicio
    
    # Guardar solo las filas con geolocalización
    with open(archivo_salida, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=nuevos_headers, delimiter=';')
        writer.writeheader()
        writer.writerows(filas_guardadas)
    
    # Resumen
    print("\n" + "=" * 60)
    print("📋 RESUMEN GEOCODIFICACIÓN")
    print("=" * 60)
    print(f"\n✅ Geocodificadas y guardadas: {geocodificadas}")
    print(f"❌ No encontradas (excluidas): {no_encontradas}")
    print(f"📊 Tasa de éxito: {geocodificadas/total_filas*100:.1f}%")
    print(f"⏱️  Tiempo total: {tiempo_total/60:.1f} minutos")
    print(f"🚀 Velocidad: {total_filas/tiempo_total:.1f} propiedades/segundo")
    print(f"\n📁 Archivo guardado: {archivo_salida}")
    print(f"📊 Total filas en archivo: {len(filas_guardadas)}")
    
    return archivo_salida


# --- Menú Principal ---
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🌍 GEOCODIFICADOR DE DIRECCIONES")
    print("=" * 60)
    
    # Listar archivos CSV disponibles (propiedades y ubicaciones, excluyendo los ya geolocalizados)
    archivos_csv = [f for f in os.listdir('.') if f.endswith('.csv') 
                    and (f.startswith('propiedades_') or f.startswith('Ubicaciones_'))
                    and '_geolocacion' not in f]
    
    if not archivos_csv:
        print("\n❌ No se encontraron archivos CSV.")
        print("   Primero ejecute el scrapper para generar datos.")
    else:
        # Ordenar por fecha
        archivos_csv.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        
        print("\n📂 Archivos CSV disponibles:\n")
        for i, archivo in enumerate(archivos_csv, 1):
            try:
                with open(archivo, 'r', encoding='utf-8-sig') as f:
                    total = sum(1 for _ in f) - 1
            except:
                total = "?"
            print(f"   [{i}] {archivo} ({total} filas)")
        
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
                        total = sum(1 for _ in f) - 1
                    
                    # Preguntar por modo
                    print("\n⚡ Seleccione el modo de geocodificación:")
                    print("   [1] 🚀 Rápido (Photon + paralelo) - Recomendado para muchos datos")
                    print("   [2] 🎯 Preciso (Nominatim + secuencial) - Más lento pero más exacto")
                    print("   [3] 🔄 ArcGIS (paralelo) - Alternativa")
                    
                    modo = input("\n👉 Seleccione modo [1]: ").strip() or "1"
                    
                    if modo == "1":
                        servicio = 'photon'
                        paralelo = True
                        tiempo_estimado = (total * DELAY_ENTRE_PETICIONES) / MAX_WORKERS / 60
                    elif modo == "2":
                        servicio = 'nominatim'
                        paralelo = False
                        tiempo_estimado = total * 1.5 / 60  # Nominatim necesita más delay
                    elif modo == "3":
                        servicio = 'arcgis'
                        paralelo = True
                        tiempo_estimado = (total * DELAY_ENTRE_PETICIONES) / MAX_WORKERS / 60
                    else:
                        servicio = 'photon'
                        paralelo = True
                        tiempo_estimado = (total * DELAY_ENTRE_PETICIONES) / MAX_WORKERS / 60
                    
                    print(f"\n⏱️  Tiempo estimado: {tiempo_estimado:.1f} minutos")
                    
                    confirmar = input("¿Continuar? (s/n): ").strip().lower()
                    
                    if confirmar == 's':
                        geocodificar_csv(archivo, usar_paralelo=paralelo, servicio=servicio)
                    else:
                        print("\n❌ Operación cancelada.")
                else:
                    print("\n❌ Opción no válida.")
                    
        except ValueError:
            print("\n❌ Entrada inválida.")
        except KeyboardInterrupt:
            print("\n\n⚠️  Operación interrumpida.")
