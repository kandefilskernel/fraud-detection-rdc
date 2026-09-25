"""Rend le paquet `ml` importable quel que soit le dossier d'où pytest est lancé."""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).absolute().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)  # les chemins ml/data/... sont relatifs à la racine du projet
