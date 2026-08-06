"""
modules/converter.py  — Protein Preparation for Docking
Convert a fixed PDB receptor to PDBQT format for AutoDock Vina.

For receptor (protein) PDBQT:
  - Keep heavy atoms + polar hydrogens (bonded to N, O, S)
  - AutoDock Vina does not require partial charges on the receptor;
    its internal scoring function handles this implicitly.
  - Atom types follow the AutoDock 4 convention.

Reference:
  Trott O & Olson AJ (2010) AutoDock Vina: improving the speed and accuracy
  of docking. J Comput Chem 31:455-461.
"""
import os, re, tempfile

# AutoDock atom type mapping (element → AD4 type)
_AD4_TYPES = {
    "C": "C",  "N": "NA", "O": "OA", "S": "SA",
    "H": "HD", "P": "P",  "F": "F",  "CL": "Cl",
    "BR": "Br","I": "I",  "FE": "Fe","ZN": "Zn",
    "MG": "Mg","CA": "Ca","MN": "Mn","CO": "Co",
    "NI": "Ni","CU": "Cu",
}

def _element_from_pdb_line(line: str) -> str:
    """Extract element symbol from PDB ATOM/HETATM line."""
    if len(line) >= 78:
        el = line[76:78].strip()
        if el: return el.upper()
    # Fall back to atom name column
    name = line[12:16].strip().lstrip("0123456789")
    return name[:2].upper() if name else "C"

def _is_polar_h(line: str, pdb_lines: list, idx: int) -> bool:
    """
    Heuristic: a hydrogen is 'polar' if the preceding heavy atom
    is N, O, or S (standard approach for receptor PDBQT preparation).
    We look backwards for the nearest ATOM line with a heavy atom.
    """
    for j in range(idx - 1, max(0, idx - 5), -1):
        prev = pdb_lines[j]
        if prev.startswith(("ATOM  ", "HETATM")):
            el = _element_from_pdb_line(prev)
            return el in ("N", "O", "S")
    return False

def pdb_to_pdbqt(pdb_text: str) -> str:
    """
    Convert a PDB string to PDBQT format suitable for AutoDock Vina receptor.
    Returns the PDBQT content as a string.
    """
    lines = pdb_text.splitlines()
    out   = ["REMARK  Converted by Protein Preparation Platform",
             "REMARK  Receptor PDBQT for AutoDock Vina",
             "REMARK  Polar hydrogens only (Trott & Olson 2010 J Comput Chem 31:455)"]

    for idx, line in enumerate(lines):
        record = line[:6].strip()

        if record not in ("ATOM", "HETATM"):
            # Keep TER and END, skip everything else
            if record in ("TER", "END"):
                out.append(line.rstrip())
            continue

        element = _element_from_pdb_line(line)

        # Skip non-polar hydrogens
        if element == "H":
            if not _is_polar_h(line, lines, idx):
                continue
            ad4 = "HD"
        else:
            ad4 = _AD4_TYPES.get(element, element[:2].capitalize())

        # Pad / truncate line to 79 chars and append AD4 type + charge placeholder
        padded = line.rstrip().ljust(79)
        # PDBQT format: columns 78-79 = partial charge (0.000), 80-82 = atom type
        pdbqt_line = padded[:66] + "  0.000" + f"  {ad4:<2}"
        out.append(pdbqt_line)

    out.append("END")
    return "\n".join(out) + "\n"

def save_pdbqt(pdb_text: str) -> str:
    """Write PDBQT to a temp file and return its path."""
    content = pdb_to_pdbqt(pdb_text)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdbqt", mode="w")
    tmp.write(content); tmp.close()
    return tmp.name, content
