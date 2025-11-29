import requests
import json
import time
from bs4 import BeautifulSoup

def scrape_page(url):
    """Función para scrapear una sola página y devolver sus resultados."""
    print(f"Scrapeando URL: {url}")
    try:
        response = requests.get(url)
        # Esto generará un error para códigos como 404, 500, etc.
        response.raise_for_status()  
        
        soup = BeautifulSoup(response.text, 'html.parser')
        script_tag = soup.find('script', {'id': '__PRELOADED_STATE__'})
        
        if not script_tag:
            print("No se encontró la etiqueta __PRELOADED_STATE__ en esta página.")
            return [], None, None

        json_data = json.loads(script_tag.string)
        
        initial_state = json_data.get('pageState', {}).get('initialState', {})
        total_results = initial_state.get('melidata_track', {}).get('event_data', {}).get('total', 0)
        items_per_page = initial_state.get('melidata_track', {}).get('event_data', {}).get('limit', 50)
        
        results = initial_state.get('results', [])
        return results, total_results, items_per_page

    # Si ocurre un error de HTTP (como un 404), la función devolverá una lista vacía
    except requests.exceptions.RequestException as e:
        print(f"Error al obtener la URL {url}: {e}")
        return [], None, None
    except Exception as e:
        print(f"Ocurrió un error procesando la página {url}: {e}")
        return [], None, None

# --- Script Principal ---
base_url = 'https://www.portalinmobiliario.com/venta/departamento/estacion-central-metropolitana'
all_properties = []

print("Obteniendo información inicial de paginación...")
results, total_results, items_per_page = scrape_page(base_url)

if total_results and items_per_page:
    print(f"Total de resultados (según el sitio): {total_results}")
    print(f"Resultados por página: {items_per_page}")

    all_properties.extend(results)

    # El límite real suele ser 2000, así que lo usamos como tope seguro
    # aunque el sitio reporte más.
    limit_real = min(total_results + 1, 2051)

    for offset in range(items_per_page + 1, limit_real, items_per_page):
        paginated_url = f"{base_url}/_Desde_{offset}_NoIndex_True"
        
        page_results, _, _ = scrape_page(paginated_url)
        
        # ---> ¡ESTA ES LA LÍNEA CLAVE! <---
        # Si la página dio un error 404 o no trajo resultados, nos detenemos.
        if not page_results:
            print("No se encontraron más resultados o se alcanzó el límite del sitio. Deteniendo el scraper.")
            break
            
        all_properties.extend(page_results)
        
        time.sleep(1)

print("\n--- Extracción Completa ---")
# Filtramos para asegurarnos de que solo procesamos tarjetas de propiedades
property_cards = [item for item in all_properties if item.get('id') == 'POLYCARD']
print(f"Total de tarjetas de propiedades procesadas: {len(property_cards)}")

print("\n--- Procesando Datos de Propiedades ---")
for item in property_cards:
    polycard = item.get('polycard', {})
    components = polycard.get('components', [])
    
    title = "No disponible"
    price = "No disponible"
    currency = ""
    attributes = []
    location = "No disponible"

    for component in components:
        if component.get('type') == 'title':
            title = component.get('title', {}).get('text')
        elif component.get('type') == 'price':
            price_info = component.get('price', {})
            price = price_info.get('current_price', {}).get('value')
            currency = price_info.get('current_price', {}).get('currency')
        elif component.get('type') == 'attributes_list':
            attributes = component.get('attributes_list', {}).get('texts')
        elif component.get('type') == 'location':
            location = component.get('location', {}).get('text')
    
    print(f"Título: {title}")
    print(f"Precio: {price} {currency if price != 'No disponible' else ''}")
    if attributes:
        print(f"Características: {', '.join(attributes)}")
    else:
        print("Características: No especificadas")
    print(f"Ubicación: {location}")
    print("-" * 20)