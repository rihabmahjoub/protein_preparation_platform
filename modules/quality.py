"""modules/quality.py — Protein Preparation for Docking"""
import numpy as np, io
from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley

BACKBONE = {"N","CA","C","O"}

def assess_quality(pdb_text:str, label:str="structure") -> dict:
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure(label, io.StringIO(pdb_text))
    chains=residues=atoms=het=water=0
    missing_bb=[]; b_factors=[]
    for model in struct:
        for chain in model:
            chains+=1
            for res in chain:
                het_flag,seq,_=res.id
                if het_flag.strip():
                    if het_flag.strip()=="W": water+=1
                    else: het+=1
                    continue
                residues+=1
                missing=BACKBONE-{a.name for a in res}
                if missing:
                    missing_bb.append({"chain":chain.id,"resnum":seq,
                                       "resname":res.resname,"missing":list(missing)})
                for atom in res:
                    atoms+=1; b_factors.append(atom.get_bfactor())
        break
    b=np.array(b_factors) if b_factors else np.array([0.])
    try:
        sr=ShrakeRupley(); sr.compute(struct,level="S")
        sasa=round(float(struct[0].sasa),1)
    except: sasa=None
    return {"label":label,"n_chains":chains,"n_residues":residues,
            "n_atoms":atoms,"n_heterogens":het,"n_waters":water,
            "n_missing_bb_res":len(missing_bb),"missing_bb_details":missing_bb[:20],
            "backbone_completeness":round(1-len(missing_bb)/max(residues,1),4),
            "bfactor_mean":round(float(b.mean()),2),"bfactor_std":round(float(b.std()),2),
            "bfactor_max":round(float(b.max()),2),"total_sasa_A2":sasa}

def compare_quality(before:dict, after:dict) -> list:
    changes=[]
    dh=after["n_heterogens"]-before["n_heterogens"]
    dw=after["n_waters"]-before["n_waters"]
    da=after["n_atoms"]-before["n_atoms"]
    dm=after["n_missing_bb_res"]-before["n_missing_bb_res"]
    if dh<0: changes.append(f"Heterogens removed: {abs(dh)}")
    if dw<0: changes.append(f"Water molecules removed: {abs(dw)}")
    if da>0: changes.append(f"Atoms added (H + missing): +{da}")
    if dm<0: changes.append(f"Missing backbone residues reconstructed: {abs(dm)}")
    changes.append(
        f"Backbone completeness: "
        f"{before['backbone_completeness']*100:.1f}% → "
        f"{after['backbone_completeness']*100:.1f}%")
    return changes
