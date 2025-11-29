import requests
import json
from bs4 import BeautifulSoup

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
}

r = requests.get('https://www.portalinmobiliario.com/venta/departamento/estacion-central-metropolitana', headers=headers)
soup = BeautifulSoup(r.text, 'html.parser')
script = soup.find('script', {'id': '__PRELOADED_STATE__'})
data = json.loads(script.string)
results = data.get('pageState', {}).get('initialState', {}).get('results', [])

print(f"Total resultados: {len(results)}")

# Buscar el primer POLYCARD
for r in results:
    if r.get('id') == 'POLYCARD':
        print("\n--- Estructura de POLYCARD ---")
        print(f"Keys principales: {r.keys()}")
        
        polycard = r.get('polycard', {})
        print(f"\nKeys en polycard: {polycard.keys()}")
        
        components = polycard.get('components', [])
        print(f"\nComponentes ({len(components)}):")
        for comp in components:
            print(f"  Tipo: {comp.get('type')}")
            if comp.get('type') == 'title':
                print(f"    -> Title: {comp}")
            elif comp.get('type') == 'price':
                print(f"    -> Price: {comp}")
            elif comp.get('type') == 'attributes_list':
                print(f"    -> Attrs: {comp}")
            elif comp.get('type') == 'location':
                print(f"    -> Location: {comp}")
        
        # Verificar tracks para URL
        tracks = polycard.get('tracks', {})
        if tracks:
            print(f"\nTracks: {tracks.keys()}")
            melidata = tracks.get('melidata_track', {})
            if melidata:
                event_data = melidata.get('event_data', {})
                print(f"URL: {event_data.get('url')}")
        break
