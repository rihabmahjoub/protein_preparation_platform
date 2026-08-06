"""
modules/protonation.py — Protein Preparation for Docking
pKa-based protonation state analysis at target pH.
References:
  Lehninger AL (2017) Principles of Biochemistry, 7th ed. W.H. Freeman.
  Pace CN et al. (2009) PNAS 106:2665-2670. [Cys pKa]
  Tanford C (1962) Adv Protein Chem 17:69-165. [His pKa]
"""

PKA_TABLE = {
    "HIS": (6.0,  "HIP (doubly protonated)",    "HIE/HID (singly protonated)"),
    "CYS": (8.3,  "CYS-SH (protonated thiol)",  "CYS-S⁻ (thiolate)"),
    "ASP": (3.9,  "ASP-COOH (protonated)",       "ASP-COO⁻ (carboxylate)"),
    "GLU": (4.1,  "GLU-COOH (protonated)",       "GLU-COO⁻ (carboxylate)"),
    "TYR": (10.1, "TYR-OH (protonated phenol)",  "TYR-O⁻ (phenolate)"),
    "LYS": (10.5, "LYS-NH₃⁺ (protonated)",      "LYS-NH₂ (free amine)"),
    "ARG": (12.5, "ARG-guanidinium⁺",            "ARG-guanidine (neutral)"),
}
STANDARD_STATE_PH74 = {
    "HIS": "HIE/HID (neutral, ~96% deprotonated at pH 7.4)",
    "CYS": "CYS-SH (neutral, ~89% protonated at pH 7.4)",
    "ASP": "ASP-COO⁻ (negatively charged)",
    "GLU": "GLU-COO⁻ (negatively charged)",
    "TYR": "TYR-OH (neutral)",
    "LYS": "LYS-NH₃⁺ (positively charged)",
    "ARG": "ARG-guanidinium⁺ (positively charged)",
}
AMBIGUITY_WINDOW = 2.0
TITRATABLE = set(PKA_TABLE.keys())


def fraction_protonated(pka: float, ph: float) -> float:
    return 1.0 / (1.0 + 10 ** (ph - pka))


def analyze_protonation(structure, ph: float = 7.4) -> dict:
    results, ambiguous = [], []
    for model in structure:
        for chain in model:
            for res in chain:
                rname = res.resname.strip()
                if rname not in TITRATABLE:
                    continue
                pka, acid_form, base_form = PKA_TABLE[rname]
                fp    = fraction_protonated(pka, ph)
                delta = abs(ph - pka)
                is_ambig = delta < AMBIGUITY_WINDOW
                entry = {
                    "chain":              chain.id,
                    "resnum":             res.id[1],
                    "resname":            rname,
                    "pka":                pka,
                    "delta_pH_pKa":       round(delta, 2),
                    "frac_protonated":    round(fp, 3),
                    "frac_deprotonated":  round(1 - fp, 3),
                    "predominant_state":  acid_form if fp > 0.5 else base_form,
                    "standard_state":     STANDARD_STATE_PH74.get(rname, ""),
                    "ambiguous":          is_ambig,
                }
                results.append(entry)
                if is_ambig:
                    ambiguous.append(entry)
        break

    type_counts = {}
    for r in results:
        type_counts[r["resname"]] = type_counts.get(r["resname"], 0) + 1

    recs = []
    his = [r for r in ambiguous if r["resname"] == "HIS"]
    cys = [r for r in ambiguous if r["resname"] == "CYS"]

    if his:
        pos = ", ".join(f"{r['chain']}{r['resnum']}" for r in his)
        recs.append(
            f"HISTIDINE ({pos}): pKa ~6.0 is 1.4 units from pH {ph}. "
            f"HIE (Nε-protonated) is the standard default assignment, but "
            f"HID (Nδ-protonated) or HIP (doubly protonated) may significantly "
            f"affect docking if these residues are near the binding site. "
            f"Verify manually or use PROPKA for local pKa prediction."
        )
    if cys:
        pos = ", ".join(f"{r['chain']}{r['resnum']}" for r in cys)
        recs.append(
            f"CYSTEINE ({pos}): pKa ~8.3, 0.9 units from pH {ph}. "
            f"~89% protonated as CYS-SH at pH {ph}. If located in the active "
            f"site, the thiolate form (CYS-S⁻) may be catalytically relevant. "
            f"Check whether this is a catalytic cysteine or part of a disulfide bond."
        )
    if not recs:
        recs.append(
            f"No ambiguous residues detected at pH {ph}. "
            f"Standard protonation states have been applied."
        )
    recs.append(
        "For structure-aware pKa prediction accounting for local electrostatics, "
        "use PROPKA (Li H et al. 2005 Proteins 61:704) or the H++ server."
    )

    return {
        "ph":              ph,
        "residues":        results,
        "ambiguous":       ambiguous,
        "type_counts":     type_counts,
        "n_total":         len(results),
        "n_ambiguous":     len(ambiguous),
        "recommendations": recs,
    }
