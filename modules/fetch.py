"""
modules/fetch.py  — Protein Preparation for Docking
Fetch experimental structures from the RCSB Protein Data Bank.
API reference: https://data.rcsb.org/
"""
import requests

RCSB_PDB  = "https://files.rcsb.org/download/{pdb_id}.pdb"
RCSB_META = "https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"

def fetch_pdb(pdb_id: str) -> str:
    """Download a PDB file from RCSB."""
    url = RCSB_PDB.format(pdb_id=pdb_id.upper())
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.text

def fetch_metadata(pdb_id: str) -> dict:
    """Fetch entry metadata (resolution, method, organism, title)."""
    url = RCSB_META.format(pdb_id=pdb_id.upper())
    r = requests.get(url, timeout=15)
    if not r.ok:
        return {}
    d = r.json()
    struct_info = d.get("struct", {})
    refine      = (d.get("refine") or [{}])[0]
    exptl       = (d.get("exptl") or [{}])[0]
    entity      = (d.get("polymer_entities") or [{}])
    organism    = ""
    if entity:
        sources = entity[0].get("rcsb_entity_source_organism") or [{}]
        organism = (sources[0] or {}).get("ncbi_scientific_name", "")
    return {
        "title":      struct_info.get("title", "N/A"),
        "method":     exptl.get("method", "N/A"),
        "resolution": refine.get("ls_d_res_high", "N/A"),
        "organism":   organism,
        "pdb_id":     pdb_id.upper(),
    }

def read_uploaded(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="replace")
