"""
modules/pocket.py — Protein Preparation for Docking
Binding pocket detection using a grid-based geometric algorithm
(LIGSITE-style PSP scan, following Hendlich et al. 1997 / SURFNET).

The protein's heavy atoms are mapped onto a 3D voxel grid using their
realistic van der Waals radii + a standard 1.4 Å water probe. For every
free (solvent) voxel, the algorithm scans outward along 7 principal
directions (the 3 axes + 4 cube diagonals). A direction registers a
"protein-solvent-protein" (PSP) event if protein is found within
MAX_DIST_A on BOTH sides of the voxel along that line. A voxel is
flagged as part of a pocket if at least MIN_DIRECTIONS of the 7 scan
lines register a PSP event.

This is a *local* enclosure test, not a global flood-fill/topological
one. That distinction matters: many real binding sites — including most
kinase ATP pockets — are wide, shallow clefts whose mouths are open to
bulk solvent. A global "enclosed cavity" test (flood-fill from the grid
boundary, or morphological closing of the protein mask) systematically
misses these, because any single closing radius large enough to seal a
wide mouth also fills in the pocket itself when the mouth is wide
relative to the pocket's own depth — which is the case for most real
drug-binding clefts. The PSP scan avoids this failure mode entirely: it
never requires topological enclosure, only that the protein "frames" the
voxel from multiple directions within a plausible pocket-scale distance.

References:
  Hendlich M, Rippmann F, Barnickel G (1997) LIGSITE. J Mol Graph Model 15:359-363.
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

PROBE          = 1.4   # Å — standard water probe, used to build an accurate
                        #     solvent-excluded protein mask
MAX_DIST_A     = 12.0  # Å — how far the PSP scan looks for protein along
                        #     each of the 7 directions. Raise to catch wider
                        #     clefts; lower to require tighter enclosure.
MIN_DIRECTIONS = 4     # out of 7 scan lines that must show a PSP event for
                        #     a voxel to count as "pocket". Lower = more
                        #     permissive (more, shallower pockets); higher =
                        #     stricter (fewer, more enclosed pockets).

# 3 axes + 4 cube diagonals — the 7 principal directions used by LIGSITE.
_DIRECTIONS = [
    (1,0,0), (0,1,0), (0,0,1),
    (1,1,1), (1,1,-1), (1,-1,1), (1,-1,-1),
]

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


def _shift(mask, direction, step):
    """Return `mask` shifted by `step` voxels along `direction` (zero-padded)."""
    out = np.zeros_like(mask)
    src, dst = [], []
    for d, n in zip(direction, mask.shape):
        s = d * step
        if s > 0:
            src.append(slice(s, n));    dst.append(slice(0, n - s))
        elif s < 0:
            src.append(slice(0, n + s)); dst.append(slice(-s, n))
        else:
            src.append(slice(0, n));    dst.append(slice(0, n))
    out[tuple(dst)] = mask[tuple(src)]
    return out


def _first_hit_within(protein, direction, max_steps):
    """
    For every voxel, is `protein` reached within `max_steps` when scanning
    along `direction`? Vectorized: accumulate shifted copies of `protein`
    one step at a time and record the first hit.
    """
    hit = np.zeros(protein.shape, dtype=bool)
    for step in range(1, max_steps + 1):
        hit |= _shift(protein, direction, step)
    return hit


def detect_pockets(structure, resolution:float=1.0,
                   min_volume:float=150.0, n_top:int=5,
                   max_dist_A:float=MAX_DIST_A,
                   min_directions:int=MIN_DIRECTIONS) -> list:
    """
    Detect and rank binding pockets in a prepared protein structure.

    Parameters
    ----------
    resolution     : grid spacing in Å. Smaller = more accurate but slower.
                      Recommended: 1.0 Å for standard proteins, 1.5 Å for large ones.
    min_volume     : minimum pocket volume in Å³ to report.
                      Increase to filter false positives; decrease to find small pockets.
    n_top          : maximum number of pockets to return, ranked by druggability score.
    max_dist_A     : Å — how far the PSP scan looks for protein along each of the
                      7 directions. Widen for large, shallow clefts; narrow to
                      require tighter, more classically "buried" pockets.
    min_directions : out of 7 scan directions that must register a PSP event.
                      Lower is more permissive (catches shallower/open clefts,
                      at the cost of more surface noise); higher is stricter.

    Returns
    -------
    list of dicts sorted by druggability_score (descending).

    Notes on detection limits
    --------------------------
    This is a local PSP (protein-solvent-protein) scan, not a global
    flood-fill enclosure test — it will find wide, shallow clefts (e.g.
    most kinase ATP pockets) that a topological-enclosure method would
    miss. It can still under-report very large, gently curved binding
    surfaces (e.g. some protein-protein interaction grooves) if fewer
    than `min_directions` scan lines find protein within `max_dist_A`;
    for those, try raising `max_dist_A` and/or lowering `min_directions`.
    """
    coords, elements, residues = _heavy_atoms(structure)
    if len(coords) == 0:
        return []

    radii = np.array([VDW.get(el, 1.70) + PROBE for el in elements])

    margin = max(6.0, max_dist_A)
    lo    = coords.min(axis=0) - margin
    hi    = coords.max(axis=0) + margin
    shape = (np.ceil((hi - lo) / resolution)).astype(int) + 1

    # ── protein mask (realistic radii, standard solvent probe only) ────────────
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

    # ── LIGSITE-style PSP scan ───────────────────────────────────────────────
    max_steps = int(np.ceil(max_dist_A / resolution))
    psp_count = np.zeros(shape, dtype=np.int8)
    for d in _DIRECTIONS:
        dp, dm = np.array(d), -np.array(d)
        hit_plus  = _first_hit_within(protein, tuple(dp), max_steps)
        hit_minus = _first_hit_within(protein, tuple(dm), max_steps)
        psp_count += (hit_plus & hit_minus).astype(np.int8)

    pocket_mask = (~protein) & (psp_count >= min_directions)

    poc_lab, n_poc = ndimage.label(
        pocket_mask, structure=ndimage.generate_binary_structure(3, 3)
    )
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
        # Buriedness: average fraction of the 7 scan directions that hit
        # protein on both sides, over this pocket's voxels (0 = barely
        # enclosed, 1 = enclosed from every direction).
        buried = float(psp_count[pmask].mean()) / len(_DIRECTIONS)
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
