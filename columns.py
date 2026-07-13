from __future__ import annotations

COORD_COLUMNS = ["X", "Y", "Z"]
TARGET_COLUMNS = ["AS", "S", "CORG-1", "CA", "FE"]
AU_COL = "Au_Final"

CHEMICAL_CANDIDATES = [
    "Au_Final",
    "Ag (ME-ICP61),ppm",
    "Al (ME-ICP61),%",
    "AS",
    "Ba (ME-ICP61),ppm",
    "Be (ME-ICP61),ppm",
    "Bi (ME-ICP61),ppm",
    "CA",
    "Cd (ME-ICP61),ppm",
    "Co (ME-ICP61),ppm",
    "Cr (ME-ICP61),ppm",
    "Cu (ME-ICP61),ppm",
    "FE",
    "K (ME-ICP61),%",
    "La (ME-ICP61),ppm",
    "Li (ME-ICP61),ppm",
    "Mg (ME-ICP61),%",
    "Mn (ME-ICP61),ppm",
    "Mo (ME-ICP61),ppm",
    "Na (ME-ICP61),%",
    "Ni (ME-ICP61),ppm",
    "P (ME-ICP61),ppm",
    "Pb (ME-ICP61),ppm",
    "Sb (ME-ICP61),ppm",
    "Sc (ME-ICP61),ppm",
    "Sn (ME-ICP61),ppm",
    "Sr (ME-ICP61),ppm",
    "Ti (ME-ICP61),%",
    "V (ME-ICP61),ppm",
    "W (ME-ICP61),ppm",
    "Y (ME-ICP61),ppm",
    "Zn (ME-ICP61),ppm",
    "S",
    "CORG-1",
]

LITHOLOGY_CANDIDATES = [
    "LITH",
    "LITH_STRUCTURE",
    "LITH_TEXTURE",
    "LITH_COLOUR",
    "LITH_INCLUSIONS",
    "LITH_VEIN",
    "REDOX",
    "LITH2",
    "LITH1_VOL",
    "COLOUR_2",
    "ALTERATION",
    "MINERAL_GP",
    "TYPE_MINERAL_GP",
]

BLOCK_NUMERIC_CANDIDATES = [
    *COORD_COLUMNS,
    "_X",
    "_Y",
    "_Z",
    "Au_Final",
    "volume",
    "DENSITY",
    "RESCAT",
    "ZONE",
    "PVALUE",
    "IND",
    "RESCAT_C",
]

BLOCK_CATEGORICAL_CANDIDATES = ["domain", "MODAREA", "MINED"]


def normalize_column_map() -> dict[str, str]:
    return {
        "Sобщ (S-IR08),%": "S",
        "S (ME-ICP61),%": "S_ME_ICP61",
        "As (ME-ICP61),ppm": "AS",
        "Ca (ME-ICP61),%": "CA",
        "Fe (ME-ICP61),%": "FE",
        "X_fact": "X",
        "Y_fact": "Y",
        "Z_fact": "Z",
        "EAST": "X",
        "NORTH": "Y",
        "RL": "Z",
        "AU": "Au_Final",
        "C organic (C-IR06),%": "CORG-1",
        "C organic-2 (C-IR06),%": "CORG-2",
        "CORG": "CORG-1",
    }
