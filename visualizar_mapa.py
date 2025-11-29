"""
Script para visualizar propiedades geolocalizadas en un mapa interactivo.
Muestra los puntos de las propiedades y los bordes de la comuna.
"""

import csv
import os
import webbrowser

try:
    import folium
except ImportError:
    print("❌ Instalando folium...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'folium'])
    import folium

try:
    import requests
except ImportError:
    print("❌ Instalando requests...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'requests'])
    import requests


def obtener_geojson_comuna(nombre_comuna):
    """
    Intenta obtener el GeoJSON de los límites de una comuna.
    Usa la API de Nominatim de OpenStreetMap.
    """
    try:
        url = f"https://nominatim.openstreetmap.org/search"
        params = {
            'q': f"{nombre_comuna}, Santiago, Chile",
            'format': 'json',
            'polygon_geojson': 1,
            'limit': 1
        }
        headers = {'User-Agent': 'PortalInmobiliarioScraper/1.0'}
        
        response = requests.get(url, params=params, headers=headers, timeout=10)
        data = response.json()
        
        if data and len(data) > 0:
            geojson = data[0].get('geojson')
            if geojson:
                return geojson
        
        return None
    except Exception as e:
        print(f"⚠️  No se pudo obtener el borde de la comuna: {e}")
        return None


def leer_propiedades_geolocalizadas(archivo):
    """Lee el archivo CSV y extrae las propiedades con coordenadas."""
    propiedades = []
    comunas = set()
    
    with open(archivo, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f, delimiter=';')
        for fila in reader:
            lat = fila.get('latitud', '')
            lng = fila.get('longitud', '')
            
            if lat and lng:
                try:
                    propiedades.append({
                        'lat': float(lat),
                        'lng': float(lng),
                        'titulo': fila.get('titulo', 'Sin título'),
                        'precio': fila.get('precio', 'N/A'),
                        'moneda': fila.get('moneda', ''),
                        'dormitorios': fila.get('dormitorios', ''),
                        'banos': fila.get('banos', ''),
                        'metros': fila.get('metros_utiles', ''),
                        'ubicacion': fila.get('ubicacion', ''),
                        'tipo': fila.get('tipo_propiedad', ''),
                        'comuna': fila.get('comuna', '')
                    })
                    
                    comuna = fila.get('comuna', '').strip()
                    if comuna:
                        comunas.add(comuna)
                except ValueError:
                    continue
    
    return propiedades, list(comunas)


def crear_mapa(propiedades, comunas, archivo_salida='mapa_propiedades.html'):
    """Crea un mapa interactivo con las propiedades y bordes de comunas."""
    
    if not propiedades:
        print("❌ No hay propiedades para mostrar.")
        return None
    
    # Calcular centro del mapa basado en las propiedades
    lat_promedio = sum(p['lat'] for p in propiedades) / len(propiedades)
    lng_promedio = sum(p['lng'] for p in propiedades) / len(propiedades)
    
    # Crear mapa base
    mapa = folium.Map(
        location=[lat_promedio, lng_promedio],
        zoom_start=14,
        tiles='OpenStreetMap'
    )
    
    # Agregar bordes de las comunas
    print(f"\n🗺️  Obteniendo bordes de comunas: {', '.join(comunas)}")
    for comuna in comunas:
        geojson = obtener_geojson_comuna(comuna)
        if geojson:
            folium.GeoJson(
                geojson,
                name=f'Límites {comuna}',
                style_function=lambda x: {
                    'fillColor': '#3388ff',
                    'color': '#0000ff',
                    'weight': 3,
                    'fillOpacity': 0.1
                }
            ).add_to(mapa)
            print(f"   ✅ Borde agregado: {comuna}")
        else:
            print(f"   ⚠️  No se encontró borde para: {comuna}")
    
    # Colores por tipo de propiedad
    colores = {
        'departamento': 'blue',
        'casa': 'green',
        'default': 'red'
    }
    
    # Agregar marcadores de propiedades
    print(f"\n📍 Agregando {len(propiedades)} propiedades al mapa...")
    
    for prop in propiedades:
        # Determinar color
        tipo = prop['tipo'].lower() if prop['tipo'] else 'default'
        color = colores.get(tipo, colores['default'])
        
        # Crear popup con información
        precio_str = f"{prop['precio']} {prop['moneda']}" if prop['precio'] != 'N/A' else 'Precio no disponible'
        
        popup_html = f"""
        <div style="width: 250px;">
            <h4 style="margin: 0 0 10px 0; color: #333;">{prop['titulo'][:50]}...</h4>
            <p style="margin: 5px 0;"><b>💰 Precio:</b> {precio_str}</p>
            <p style="margin: 5px 0;"><b>🏠 Tipo:</b> {prop['tipo'].capitalize() if prop['tipo'] else 'N/A'}</p>
            <p style="margin: 5px 0;"><b>🛏️ Dormitorios:</b> {prop['dormitorios']}</p>
            <p style="margin: 5px 0;"><b>🚿 Baños:</b> {prop['banos']}</p>
            <p style="margin: 5px 0;"><b>📐 Metros:</b> {prop['metros']}</p>
            <p style="margin: 5px 0;"><b>📍 Ubicación:</b> {prop['ubicacion'][:60]}...</p>
        </div>
        """
        
        folium.CircleMarker(
            location=[prop['lat'], prop['lng']],
            radius=8,
            popup=folium.Popup(popup_html, max_width=300),
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            weight=2
        ).add_to(mapa)
    
    # Agregar leyenda
    leyenda_html = """
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; 
                background-color: white; padding: 10px; border-radius: 5px;
                border: 2px solid grey; font-size: 14px;">
        <p style="margin: 0 0 5px 0;"><b>Leyenda:</b></p>
        <p style="margin: 2px 0;"><span style="color: blue;">●</span> Departamento</p>
        <p style="margin: 2px 0;"><span style="color: green;">●</span> Casa</p>
        <p style="margin: 2px 0;"><span style="color: #0000ff;">▬</span> Límite comuna</p>
    </div>
    """
    mapa.get_root().html.add_child(folium.Element(leyenda_html))
    
    # Agregar control de capas
    folium.LayerControl().add_to(mapa)
    
    # Guardar mapa
    mapa.save(archivo_salida)
    print(f"\n✅ Mapa guardado: {archivo_salida}")
    
    return archivo_salida


# --- Menú Principal ---
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🗺️  VISUALIZADOR DE PROPIEDADES EN MAPA")
    print("=" * 60)
    
    # Listar archivos geolocalizados
    archivos_csv = [f for f in os.listdir('.') if f.endswith('.csv') and '_geolocacion' in f]
    
    if not archivos_csv:
        print("\n❌ No se encontraron archivos geolocalizados.")
        print("   Primero ejecute el geocodificador para generar datos con coordenadas.")
    else:
        # Ordenar por fecha
        archivos_csv.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        
        print("\n📂 Archivos geolocalizados disponibles:\n")
        for i, archivo in enumerate(archivos_csv, 1):
            try:
                with open(archivo, 'r', encoding='utf-8-sig') as f:
                    total = sum(1 for _ in f) - 1
            except:
                total = "?"
            print(f"   [{i}] {archivo} ({total} propiedades)")
        
        print(f"\n   [0] Salir")
        
        try:
            opcion = input("\n👉 Seleccione el archivo a visualizar: ").strip()
            
            if opcion == '0':
                print("\n👋 ¡Hasta luego!")
            else:
                idx = int(opcion) - 1
                if 0 <= idx < len(archivos_csv):
                    archivo = archivos_csv[idx]
                    
                    print(f"\n📂 Procesando: {archivo}")
                    
                    # Leer propiedades
                    propiedades, comunas = leer_propiedades_geolocalizadas(archivo)
                    
                    print(f"📊 Propiedades encontradas: {len(propiedades)}")
                    print(f"🏘️  Comunas: {', '.join(comunas)}")
                    
                    if propiedades:
                        # Generar nombre del mapa
                        nombre_base = os.path.splitext(archivo)[0]
                        archivo_mapa = f"{nombre_base}_mapa.html"
                        
                        # Crear mapa
                        resultado = crear_mapa(propiedades, comunas, archivo_mapa)
                        
                        if resultado:
                            # Abrir en navegador
                            abrir = input("\n¿Abrir mapa en el navegador? (s/n): ").strip().lower()
                            if abrir == 's':
                                ruta_completa = os.path.abspath(archivo_mapa)
                                webbrowser.open(f'file://{ruta_completa}')
                                print(f"🌐 Abriendo mapa en navegador...")
                    else:
                        print("\n❌ No hay propiedades con coordenadas en el archivo.")
                else:
                    print("\n❌ Opción no válida.")
                    
        except ValueError:
            print("\n❌ Entrada inválida.")
        except KeyboardInterrupt:
            print("\n\n⚠️  Operación cancelada.")
