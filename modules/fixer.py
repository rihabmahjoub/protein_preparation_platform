"""
modules/fixer.py — Protein Preparation for Docking
Structure fixing pipeline using PDBFixer (Eastman et al. 2013).
Steps: remove solvent/ligands → add missing residues → add missing heavy atoms
       → add hydrogens at target pH → write fixed PDB.

Reference:
  Eastman P et al. (2013) OpenMM 4. J Chem Theory Comput 9:461-469.
"""
import os, tempfile

def fix_structure(pdb_text: str,
                  remove_heterogens: bool = True,
                  remove_water: bool = True,
                  add_missing_residues: bool = True,
                  add_missing_atoms: bool = True,
                  add_hydrogens: bool = True,
                  ph: float = 7.4) -> tuple:
    try:
        from pdbfixer import PDBFixer
        from openmm.app import PDBFile
    except ImportError:
        raise ImportError(
            "pdbfixer / openmm not installed. "
            "Install via: conda install -c conda-forge pdbfixer openmm"
        )

    rep = {"steps": [], "warnings": []}

    tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix=".pdb", mode="w")
    tmp_in.write(pdb_text); tmp_in.close()

    fixer = PDBFixer(filename=tmp_in.name)
    os.unlink(tmp_in.name)

    n_res_orig  = fixer.topology.getNumResidues()
    n_atoms_orig = fixer.topology.getNumAtoms()
    rep["original"] = {
        "chains":   fixer.topology.getNumChains(),
        "residues": n_res_orig,
        "atoms":    n_atoms_orig,
    }

    if remove_heterogens:
        fixer.removeHeterogens(keepWater=not remove_water)
        rep["steps"].append("Heterogens (ligands) removed.")
    if remove_water:
        rep["steps"].append("Water molecules removed.")

    if add_missing_residues:
        fixer.findMissingResidues()
        n_missing = sum(len(v) for v in fixer.missingResidues.values())
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        rep["steps"].append(f"{n_missing} missing residue(s) reconstructed.")
        rep["missing_residues"] = n_missing
    elif add_missing_atoms:
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        rep["steps"].append("Missing heavy atoms added.")

    if add_hydrogens:
        fixer.addMissingHydrogens(ph)
        rep["steps"].append(f"Hydrogen atoms added (pH {ph}).")

    tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".pdb")
    tmp_out.close()
    with open(tmp_out.name, "w") as fh:
        PDBFile.writeFile(fixer.topology, fixer.positions, fh)
    with open(tmp_out.name, "r") as fh:
        fixed_text = fh.read()
    os.unlink(tmp_out.name)

    n_atoms_fixed = fixed_text.count("\nATOM")
    rep["fixed"] = {"atoms": n_atoms_fixed, "ph": ph}
    rep["steps"].append(f"Prepared structure: {n_atoms_fixed} atoms.")

    return fixed_text, rep
