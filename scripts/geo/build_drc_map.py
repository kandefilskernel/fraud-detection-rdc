"""Contours des 26 provinces de la RDC (geoBoundaries gbOpen COD ADM1, CC BY 4.0)
-> chemins SVG simplifiés en préservant les frontières communes (arcs partagés).

Usage : python scripts/geo/build_drc_map.py geoBoundaries-COD-ADM1_simplified.geojson
Source : https://www.geoboundaries.org (COD, ADM1, gbOpen, version simplifiée)."""
import json, math, unicodedata

import sys, pathlib
SRC = sys.argv[1] if len(sys.argv) > 1 else "geoBoundaries-COD-ADM1_simplified.geojson"
OUT = pathlib.Path(__file__).resolve().parents[2] / "frontend/lib/drc-provinces.ts"
TOL = 0.035          # tolérance Douglas-Peucker en degrés (~4 km)
W = 1000             # largeur du viewBox

NAMES = {  # nom geoBoundaries -> nom utilisé dans les données / affiché
    "Upper Uele": "Haut-Uele", "Lower Uele": "Bas-Uele", "North Kivu": "Nord-Kivu",
    "South Kivu": "Sud-Kivu", "Central Kasai": "Kasai-Central", "Équateur": "Équateur",
}
CITIES = {  # chef-lieu : (lon, lat)
    "Kinshasa": ("Kinshasa", 15.322, -4.325), "Haut-Katanga": ("Lubumbashi", 27.479, -11.664),
    "Nord-Kivu": ("Goma", 29.221, -1.679), "Sud-Kivu": ("Bukavu", 28.857, -2.508),
    "Kongo-Central": ("Matadi", 13.459, -5.817), "Kasai-Oriental": ("Mbuji-Mayi", 23.590, -6.136),
    "Lualaba": ("Kolwezi", 25.473, -10.715), "Tshopo": ("Kisangani", 25.191, 0.515),
    "Ituri": ("Bunia", 30.252, 1.566), "Tanganyika": ("Kalemie", 29.194, -5.947),
    "Haut-Lomami": ("Kamina", 24.990, -8.736), "Lomami": ("Kabinda", 24.481, -6.138),
    "Sankuru": ("Lusambo", 23.444, -4.973), "Kasai-Central": ("Kananga", 22.418, -5.896),
    "Kasai": ("Tshikapa", 20.800, -6.416), "Kwilu": ("Bandundu", 17.370, -3.316),
    "Kwango": ("Kenge", 17.000, -4.806), "Mai-Ndombe": ("Inongo", 18.280, -1.950),
    "Équateur": ("Mbandaka", 18.260, 0.048), "Tshuapa": ("Boende", 20.880, -0.217),
    "Mongala": ("Lisala", 21.513, 2.148), "Sud-Ubangi": ("Gemena", 19.772, 3.257),
    "Nord-Ubangi": ("Gbadolite", 21.005, 4.290), "Bas-Uele": ("Buta", 24.730, 2.790),
    "Haut-Uele": ("Isiro", 27.616, 2.774), "Maniema": ("Kindu", 25.920, -2.943),
}

def key(p): return (round(p[0], 6), round(p[1], 6))

def dp(pts, tol):
    if len(pts) < 3: return pts
    a, b = pts[0], pts[-1]
    dx, dy = b[0] - a[0], b[1] - a[1]
    L = math.hypot(dx, dy)
    best, idx = -1, 0
    for i in range(1, len(pts) - 1):
        px, py = pts[i]
        d = abs(dy * px - dx * py + b[0] * a[1] - b[1] * a[0]) / L if L else math.hypot(px - a[0], py - a[1])
        if d > best: best, idx = d, i
    if best <= tol: return [a, b]
    return dp(pts[:idx + 1], tol)[:-1] + dp(pts[idx:], tol)

d = json.load(open(SRC, encoding="utf-8"))
feats = []
for f in d["features"]:
    g = f["geometry"]
    polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
    name = NAMES.get(f["properties"]["shapeName"], f["properties"]["shapeName"])
    feats.append({"name": name, "iso": f["properties"]["shapeISO"],
                  "rings": [[key(p) for p in poly[0]] for poly in polys]})  # contours extérieurs

# qui utilise chaque sommet ?
owners = {}
for fi, f in enumerate(feats):
    for r in f["rings"]:
        for p in r: owners.setdefault(p, set()).add(fi)

cache = {}
def simplify_chain(chain):
    k = (chain[0], chain[-1], len(chain))
    rev = chain[0] > chain[-1]
    c = chain[::-1] if rev else chain
    ck = (c[0], c[-1], tuple(c[len(c) // 2]))
    if ck not in cache: cache[ck] = dp(c, TOL)
    s = cache[ck]
    return s[::-1] if rev else s

for f in feats:
    out = []
    for r in f["rings"]:
        r = r[:-1] if r[0] == r[-1] else r
        n = len(r)
        # points de rupture : changement d'ensemble de propriétaires
        br = [i for i in range(n) if owners[r[i]] != owners[r[i - 1]] or owners[r[i]] != owners[r[(i + 1) % n]]]
        if not br:
            s = dp(r + [r[0]], TOL)
            if len(s) < 4:  # petit îlot : garder au moins un triangle
                s = r[:: max(1, n // 4)] + [r[0]]
        else:
            s = []
            for j, i0 in enumerate(br):
                i1 = br[(j + 1) % len(br)]
                chain = [r[k % n] for k in range(i0, (i1 if i1 > i0 else i1 + n) + 1)]
                s += simplify_chain(chain)[:-1]
            s.append(s[0])
        if len(s) >= 4: out.append(s)
    f["rings"] = out

lons = [p[0] for f in feats for r in f["rings"] for p in r]
lats = [p[1] for f in feats for r in f["rings"] for p in r]
lon0, lon1, lat0, lat1 = min(lons), max(lons), min(lats), max(lats)
kx = math.cos(math.radians((lat0 + lat1) / 2))
sx = W / ((lon1 - lon0) * kx)
H = round((lat1 - lat0) * sx)
proj = lambda lon, lat: (round((lon - lon0) * kx * sx, 1), round((lat1 - lat) * sx, 1))

def area_centroid(ring):
    a = cx = cy = 0
    for (x0, y0), (x1, y1) in zip(ring, ring[1:]):
        c = x0 * y1 - x1 * y0; a += c; cx += (x0 + x1) * c; cy += (y0 + y1) * c
    return a / 2, (cx / (3 * a), cy / (3 * a)) if a else ring[0]

res = []
for f in sorted(feats, key=lambda f: f["name"]):
    pr = [[proj(*p) for p in r] for r in f["rings"]]
    path = "".join("M" + "L".join(f"{x:g},{y:g}" for x, y in r[:-1]) + "Z" for r in pr)
    big = max(pr, key=lambda r: abs(area_centroid(r)[0]))
    lx, ly = area_centroid(big)[1]
    city = CITIES[f["name"]]
    res.append({"name": f["name"], "iso": f["iso"], "d": path,
                "label": [round(lx, 1), round(ly, 1)],
                "city": {"name": city[0], "xy": list(proj(city[1], city[2]))}})

ts = ("// Généré par scripts/geo/build_drc_map.py — NE PAS MODIFIER À LA MAIN.\n"
      "// Contours : geoBoundaries gbOpen COD ADM1 (CC BY 4.0, www.geoboundaries.org), simplifiés\n"
      "// (Douglas-Peucker, frontières communes préservées). Projection équirectangulaire.\n\n"
      "export type Province = { name: string; iso: string; d: string; label: [number, number];\n"
      "  city: { name: string; xy: [number, number] } };\n\n"
      f"export const DRC_VIEWBOX = \"0 0 {W} {H}\";\n\n"
      f"export const DRC_PROVINCES: Province[] = {json.dumps(res, ensure_ascii=False, indent=1)};\n")
open(OUT, "w", encoding="utf-8").write(ts)
print("provinces", len(res), "taille ts", len(ts) // 1024, "Ko", "viewBox", W, H)
