"""Lightweight, offline result viewer (Folium-based, no JS bundling needed)."""

from __future__ import annotations

import json
from pathlib import Path


def make_viewer_html(geojson_path: str | Path, output_path: str | Path) -> Path:
    """Generate a single-file HTML report with leaflet + the GeoJSON inlined."""
    geojson_path = Path(geojson_path)
    output_path = Path(output_path)
    if not geojson_path.exists():
        raise FileNotFoundError(geojson_path)

    fc = json.loads(geojson_path.read_text())

    # Compute a rough centre from the first feature.
    centre = [0.0, 0.0]
    for feat in fc.get("features", []):
        try:
            coords = feat["geometry"]["coordinates"]
            while isinstance(coords[0], list):
                coords = coords[0]
            centre = [coords[1], coords[0]]
            break
        except (KeyError, IndexError, TypeError):
            continue

    inline_geojson = json.dumps(fc)
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Mokashif - change report</title>
  <link rel="stylesheet"
        href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
  <style>
    html, body, #map {{ height: 100%; margin: 0; padding: 0; }}
    .legend {{ background: white; padding: 6px 10px; line-height: 1.4em; }}
  </style>
</head>
<body>
<div id="map"></div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
  const data = {inline_geojson};
  const map = L.map('map').setView([{centre[0]}, {centre[1]}], 14);
  L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',
              {{ maxZoom: 20 }}).addTo(map);
  function style(feature) {{
    const c = (feature.properties || {{}}).confidence ?? 0.5;
    const r = Math.round(255 * c);
    return {{ color: `rgb(${{r}}, 60, 60)`, weight: 1, fillOpacity: 0.45 }};
  }}
  const layer = L.geoJSON(data, {{ style, onEachFeature: (f, l) => {{
    const p = f.properties || {{}};
    l.bindPopup(`Confidence: ${{(p.confidence ?? 0).toFixed(3)}}<br>Pixels: ${{p.area_pixels ?? '?'}}`);
  }} }}).addTo(map);
  if (layer.getBounds().isValid()) map.fitBounds(layer.getBounds());
</script>
</body>
</html>
"""
    output_path.write_text(html)
    return output_path
