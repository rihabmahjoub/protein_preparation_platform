"""
app.py — Protein Preparation for Docking Platform
Pipeline: Load → Quality → PDBFixer → Protonation QC → PDBQT → Binding Sites
"""
import os,uuid,io,tempfile
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from flask import Flask,request,jsonify,send_file,render_template,session
from modules import fetch,viewer,quality,protonation as prot_mod,converter

try:
    from modules import fixer as fixer_mod; FIXER_OK=True
except: FIXER_OK=False
try:
    from modules import pocket as pocket_mod; POCKET_OK=True
except: POCKET_OK=False

app=Flask(__name__)
app.secret_key=os.environ.get("SECRET_KEY",os.urandom(32))

# ── White publication-ready matplotlib style ──────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":"white","axes.facecolor":"#FAFBFC",
    "axes.edgecolor":"#C8D3E0","axes.labelcolor":"#1A202C",
    "axes.titlecolor":"#1A202C","xtick.color":"#5A6D87",
    "ytick.color":"#5A6D87","text.color":"#1A202C",
    "legend.facecolor":"white","legend.edgecolor":"#C8D3E0",
    "font.family":"sans-serif","font.size":9,
})

_STORE:dict={}
def _empty():
    return{"pdb_raw":None,"pdb_fixed":None,"fmt":"pdb","pdb_id":"",
           "metadata":{},"structure_raw":None,"structure_fixed":None,
           "quality_before":{},"quality_after":{},"fixer_report":{},
           "pdbqt":None,"pockets":[],"protonation":{}}
def sid():
    if "sid" not in session: session["sid"]=str(uuid.uuid4())
    return session["sid"]
def get_state():
    s=sid()
    if s not in _STORE: _STORE[s]=_empty()
    return _STORE[s]
def save_state(st): _STORE[sid()]=st
def _parse(txt,label="s"):
    from Bio.PDB import PDBParser; import io as _io
    return PDBParser(QUIET=True).get_structure(label,_io.StringIO(txt))
def _b64(fig):
    import base64; buf=io.BytesIO()
    fig.savefig(buf,format="png",bbox_inches="tight",dpi=130,facecolor="white")
    plt.close(fig)
    return "data:image/png;base64,"+base64.b64encode(buf.getvalue()).decode()

@app.route("/")
def index(): return render_template("index.html",fixer_ok=FIXER_OK,pocket_ok=POCKET_OK)

@app.route("/api/fetch_pdb",methods=["POST"])
def api_fetch():
    pid=(request.json or{}).get("pdb_id","").strip().upper()
    if not pid: return jsonify({"error":"PDB ID required."}),400
    try:
        txt=fetch.fetch_pdb(pid); meta=fetch.fetch_metadata(pid)
        struct=_parse(txt,pid); qb=quality.assess_quality(txt,"before")
        st=get_state()
        st.update({"pdb_raw":txt,"pdb_id":pid,"metadata":meta,
                   "structure_raw":struct,"quality_before":qb})
        save_state(st)
        return jsonify({"msg":f"{pid} — {qb['n_residues']} residues","meta":meta,"quality":qb})
    except Exception as e: return jsonify({"error":str(e)}),500

@app.route("/api/upload_pdb",methods=["POST"])
def api_upload():
    f=request.files.get("file")
    if not f: return jsonify({"error":"No file selected."}),400
    try:
        txt=f.read().decode("utf-8",errors="replace")
        pid=os.path.splitext(f.filename)[0].upper()
        struct=_parse(txt,pid); qb=quality.assess_quality(txt,"before")
        st=get_state()
        st.update({"pdb_raw":txt,"pdb_id":pid,"metadata":{},
                   "structure_raw":struct,"quality_before":qb})
        save_state(st)
        return jsonify({"msg":f"{pid} — {qb['n_residues']} residues","meta":{},"quality":qb})
    except Exception as e: return jsonify({"error":str(e)}),500

@app.route("/api/view_raw",methods=["POST"])
def api_view_raw():
    st=get_state()
    if not st["pdb_raw"]: return jsonify({"error":"No structure loaded."}),400
    col=(request.json or{}).get("color_by","chain")
    return jsonify({"html":viewer.build_html(st["pdb_raw"],col)})

@app.route("/api/fix",methods=["POST"])
def api_fix():
    if not FIXER_OK: return jsonify({"error":"pdbfixer not available. Use Docker."}),400
    st=get_state()
    if not st["pdb_raw"]: return jsonify({"error":"Load a structure first."}),400
    b=request.json or{}
    try:
        fixed,rep=fixer_mod.fix_structure(
            st["pdb_raw"],remove_heterogens=b.get("remove_het",True),
            remove_water=b.get("remove_water",True),
            add_missing_residues=b.get("add_res",True),
            add_missing_atoms=b.get("add_atoms",True),
            add_hydrogens=b.get("add_h",True),ph=float(b.get("ph",7.4)))
        sf=_parse(fixed,"fixed"); qa=quality.assess_quality(fixed,"after")
        ch=quality.compare_quality(st["quality_before"],qa)
        st.update({"pdb_fixed":fixed,"structure_fixed":sf,"quality_after":qa,"fixer_report":rep})
        save_state(st)
        return jsonify({"msg":"Structure prepared successfully.",
                        "fix_report":rep,"quality_after":qa,"changes":ch,
                        "html":viewer.build_html(fixed,"chain")})
    except Exception as e: return jsonify({"error":str(e)}),500

@app.route("/api/protonation",methods=["POST"])
def api_protonation():
    st=get_state()
    struct=st["structure_fixed"] or st["structure_raw"]
    if not struct: return jsonify({"error":"No structure available."}),400
    ph=float((request.json or{}).get("ph",7.4))
    try:
        res=prot_mod.analyze_protonation(struct,ph=ph)
        st["protonation"]=res; save_state(st)
        df=pd.DataFrame(res["residues"]) if res["residues"] else pd.DataFrame()
        img=None
        if not df.empty:
            fig,axes=plt.subplots(1,2,figsize=(10,3.3))
            tc=res["type_counts"]
            axes[0].bar(tc.keys(),tc.values(),color="#8B1A2A",edgecolor="white",linewidth=0.5)
            axes[0].set_xlabel("Residue type"); axes[0].set_ylabel("Count")
            axes[0].set_title("Titratable residues by type")
            axes[0].spines["top"].set_visible(False); axes[0].spines["right"].set_visible(False)
            amb=[r for r in res["residues"] if r["ambiguous"]]
            if amb:
                lbs=[f"{r['resname']}{r['resnum']}" for r in amb]
                fp=[r["frac_protonated"] for r in amb]
                cols=["#C0392B" if x>0.5 else "#2980B9" for x in fp]
                axes[1].barh(lbs,fp,color=cols,edgecolor="white",linewidth=0.5)
                axes[1].axvline(0.5,color="#7F8C8D",lw=1,linestyle="--")
                axes[1].set_xlabel(f"Fraction protonated at pH {ph}")
                axes[1].set_title(f"Ambiguous residues (|pH − pKa| < 2 units)")
                axes[1].set_xlim(0,1)
                axes[1].spines["top"].set_visible(False); axes[1].spines["right"].set_visible(False)
            else:
                axes[1].text(0.5,0.5,f"No ambiguous residues at pH {ph}",
                             ha="center",va="center",transform=axes[1].transAxes,color="#7F8C8D")
                axes[1].set_title("Ambiguous residues")
            plt.tight_layout(); img=_b64(fig)
        return jsonify({"n_total":res["n_total"],"n_ambiguous":res["n_ambiguous"],
                        "ph":ph,"residues":res["residues"],"ambiguous":res["ambiguous"],
                        "recommendations":res["recommendations"],"img":img})
    except Exception as e: return jsonify({"error":str(e)}),500

@app.route("/api/convert",methods=["POST"])
def api_convert():
    st=get_state()
    src=st["pdb_fixed"] or st["pdb_raw"]
    if not src: return jsonify({"error":"No structure available."}),400
    try:
        _,pdbqt=converter.save_pdbqt(src)
        st["pdbqt"]=pdbqt; save_state(st)
        n=sum(1 for l in pdbqt.splitlines() if l.startswith("ATOM"))
        return jsonify({"msg":f"PDBQT generated — {n} ATOM records.",
                        "preview":"\n".join(pdbqt.splitlines()[:25])})
    except Exception as e: return jsonify({"error":str(e)}),500

@app.route("/api/download_pdbqt")
def dl_pdbqt():
    st=get_state()
    if not st["pdbqt"]: return jsonify({"error":"Generate PDBQT first."}),400
    return send_file(io.BytesIO(st["pdbqt"].encode()),mimetype="chemical/x-pdbqt",
                     as_attachment=True,download_name=f"{st['pdb_id'] or 'receptor'}.pdbqt")

@app.route("/api/download_fixed_pdb")
def dl_fixed():
    st=get_state(); src=st["pdb_fixed"] or st["pdb_raw"]
    if not src: return jsonify({"error":"No structure."}),400
    return send_file(io.BytesIO(src.encode()),mimetype="chemical/x-pdb",
                     as_attachment=True,download_name=f"{st['pdb_id'] or 'prepared'}_prepared.pdb")

@app.route("/api/pockets",methods=["POST"])
def api_pockets():
    if not POCKET_OK: return jsonify({"error":"scipy not available."}),400
    st=get_state()
    struct=st["structure_fixed"] or st["structure_raw"]
    if not struct: return jsonify({"error":"No structure loaded."}),400
    b=request.json or{}
    try:
        pockets=pocket_mod.detect_pockets(struct,
            resolution=float(b.get("resolution",1.0)),
            min_volume=float(b.get("min_volume",150.0)),
            n_top=int(b.get("n_top",5)))
        st["pockets"]=pockets; save_state(st)
        src=st["pdb_fixed"] or st["pdb_raw"]
        hl=[r["resnum"] for r in pockets[0]["lining_residues"]] if pockets else []
        html=viewer.build_html(src,"chain",highlight_residues=hl)
        img=None
        if pockets:
            fig,ax=plt.subplots(figsize=(7,max(2.4,len(pockets)*0.65)))
            lbs=[f"Site {i+1}  ({p['volume_A3']:.0f} Å³)" for i,p in enumerate(pockets)]
            sc=[p["druggability_score"] for p in pockets]
            cols=["#8B1A2A" if i==0 else "#C0748A" for i in range(len(pockets))]
            ax.barh(lbs[::-1],sc[::-1],color=cols[::-1],height=0.52,edgecolor="white",linewidth=0.5)
            ax.axvline(0.5,color="#E74C3C",lw=1.1,linestyle="--",label="Druggability threshold (0.5)")
            ax.set_xlabel("Druggability score (Halgren 2009)")
            ax.set_title("Predicted binding sites")
            ax.set_xlim(0,1.05); ax.legend(fontsize=8,framealpha=0.9)
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            plt.tight_layout(); img=_b64(fig)
        summary=[{k:v for k,v in p.items() if k!="lining_residues"} for p in pockets]
        lining=[p["lining_residues"] for p in pockets]
        return jsonify({"pockets":summary,"lining":lining,"html":html,"img":img})
    except Exception as e: return jsonify({"error":str(e)}),500

if __name__=="__main__":
    app.run(debug=False,host="0.0.0.0",port=5000)
