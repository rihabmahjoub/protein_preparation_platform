"""
modules/pocket.py — Protein Preparation for Docking
Binding pocket detection using a grid-based geometric algorithm.

The protein's heavy atoms are mapped onto a 3D voxel grid. Exterior
solvent is identified by flood-fill from a padded boundary. Enclosed
voxels that are not protein are candidate pockets. Connected components
are labeled and scored by volume, hydrophobic content and buriedness.

PROBE = 2.0 Å (larger than the 1.4 Å water radius) to seal partially
open pockets such as kinase ATP-binding sites, as used in SURFNET-Ref.

References:
  Laskowski RA (1995) SURFNET. J Mol Graph 13:323-330.
  Le Guilloux V et al. (2009) Fpocket. BMC Bioinformatics 10:168.
  Halgren TA (2009) Druggability. J Chem Inf Model 49:377-389.
  Bondi A (1964) VDW radii. J Phys Chem 68:441-451.
"""
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

VDW = {"C":1.70,"N":1.55,"O":1.52,"S":1.80,"H":1.20,"P":1.80,
       "F":1.47,"CL":1.75,"BR":1.85,"I":1.98,"SE":1.90,
       "FE":2.05,"ZN":1.39,"MG":1.73,"CA":1.74,"MN":1.73}
PROBE = 2.0   # Å — enlarged to seal partially open cavities

HYDROPHOBIC = {"ALA","VAL","ILE","LEU","MET","PHE","TRP","PRO","CYS"}
POLAR       = {"SER","THR","ASN","GLN","TYR","HIS","GLY"}


def _element(atom) -> str:
    el = atom.element
    if isinstance(el, str) and el.strip():
        return el.strip().upper()
    if hasattr(el, "symbol"):
        return el.symbol.upper()
    name = atom.get_name().strip().lstrip("0123456789")
    return name[:1].upper() if name else "C"


def _heavy_atoms(structure):
    coords, elements, residues = [], [], []
    for model in structure:
        for chain in model:
            for res in chain:
                for atom in res:
                    el = _element(atom)
                    if el in ("H","D"): continue
                    coords.append(atom.get_coord())
                    elements.append(el)
                    residues.append((chain.id, res.id[1], res.resname.strip()))
        break
    return np.array(coords, dtype=float), elements, residues


def detect_pockets(structure, resolution:float=1.0,
                   min_volume:float=150.0, n_top:int=5) -> list:
    """
    Detect and rank binding pockets in a prepared protein structure.

    Parameters
    ----------
    resolution  : grid spacing in Å. Smaller = more accurate but slower.
                  Recommended: 1.0 Å for standard proteins, 1.5 Å for large ones.
    min_volume  : minimum cavity volume in Å³ to report.
                  Increase to filter false positives; decrease to find small pockets.
    n_top       : maximum number of pockets to return, ranked by druggability score.

    Returns
    -------
    list of dicts sorted by druggability_score (descending).
    """
    coords, elements, residues = _heavy_atoms(structure)
    if len(coords) == 0:
        return []

    radii = np.array([VDW.get(el, 1.70) + PROBE for el in elements])

    margin = 6.0
    lo    = coords.min(axis=0) - margin
    hi    = coords.max(axis=0) + margin
    shape = (np.ceil((hi - lo) / resolution)).astype(int) + 1

    # ── protein mask ──────────────────────────────────────────────────────────
    protein = np.zeros(shape, dtype=bool)
    for ac, rad in zip(coords, radii):
        idx  = ((ac - lo) / resolution).astype(int)
        rv   = int(np.ceil(rad / resolution)) + 1
        x0=max(0,idx[0]-rv); x1=min(shape[0],idx[0]+rv+1)
        y0=max(0,idx[1]-rv); y1=min(shape[1],idx[1]+rv+1)
        z0=max(0,idx[2]-rv); z1=min(shape[2],idx[2]+rv+1)
        gx,gy,gz = np.mgrid[x0:x1,y0:y1,z0:z1]
        gpts = np.stack([gx,gy,gz],axis=-1).astype(float)*resolution+lo
        protein[x0:x1,y0:y1,z0:z1] |= np.sum((gpts-ac)**2,axis=-1) <= rad*rad

    # ── exterior detection via padded labelling ───────────────────────────────
    free   = ~protein
    padded = np.pad(free, 1, mode="constant", constant_values=True)
    plab, _= ndimage.label(padded)
    ext    = plab[0,0,0]
    pcav   = padded & (plab != ext)
    cavity = pcav[1:-1,1:-1,1:-1]

    poc_lab, n_poc = ndimage.label(cavity)
    if n_poc == 0:
        return []

    tree = cKDTree(coords)
    LINING = 5.5

    pockets = []
    for pid in range(1, n_poc+1):
        pmask  = poc_lab == pid
        n_vox  = int(pmask.sum())
        vol    = n_vox * resolution**3
        if vol < min_volume:
            continue
        vox_idx = np.argwhere(pmask)
        center  = vox_idx.mean(axis=0)*resolution+lo
        step    = max(1, len(vox_idx)//600)
        sample  = vox_idx[::step]*resolution+lo
        lining_atoms = set()
        for pt in sample:
            lining_atoms.update(tree.query_ball_point(pt, LINING))
        lining_res = {}
        for ai in lining_atoms:
            ch,rnum,rname = residues[ai]
            lining_res[(ch,rnum)] = rname
        hf = sum(1 for rn in lining_res.values() if rn in HYDROPHOBIC)/max(len(lining_res),1)
        dilated = ndimage.binary_dilation(pmask)
        surf    = int((dilated & (plab[1:-1,1:-1,1:-1]==ext)).sum())
        buried  = 1.0 - min(surf/max(n_vox,1), 1.0)
        vs = float(np.exp(-((vol-700)/450)**2))
        ds = round(0.40*vs + 0.35*(hf**1.5) + 0.25*buried, 3)
        pockets.append({
            "id":pid,
            "center_x":round(float(center[0]),2),
            "center_y":round(float(center[1]),2),
            "center_z":round(float(center[2]),2),
            "volume_A3":round(vol,1),
            "hydrophobic_frac":round(hf,3),
            "buriedness":round(buried,3),
            "druggability_score":ds,
            "n_lining_residues":len(lining_res),
            "lining_residues":[
                {"chain":ch,"resnum":rnum,"resname":rname,
                 "class":"hydrophobic" if rname in HYDROPHOBIC
                 else "polar" if rname in POLAR else "charged"}
                for (ch,rnum),rname in sorted(lining_res.items())
            ],
        })
    pockets.sort(key=lambda x:-x["druggability_score"])
    return pockets[:n_top]
