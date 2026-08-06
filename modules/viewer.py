"""modules/viewer.py — Protein Preparation for Docking"""
import py3Dmol

_CSS = """<style>
html,body{margin:0;padding:0;width:100%;height:100%;overflow:hidden;background:white}
.mol-container,.viewer_3Dmoljs{width:100%!important;height:100%!important}
canvas{width:100%!important;height:100%!important;display:block}
</style>"""

def build_html(pdb_text:str, color_by:str="chain",
               highlight_residues=None, fmt:str="pdb",
               bg_color:str="white") -> str:
    view = py3Dmol.view(width=820, height=500)
    view.addModel(pdb_text, fmt)
    view.setBackgroundColor(bg_color)
    if color_by=="chain":
        view.setStyle({"cartoon":{"colorscheme":"chain"}})
    elif color_by=="bfactor":
        view.setStyle({"cartoon":{"colorscheme":{"prop":"b","gradient":"rwb","min":0,"max":100}}})
    else:
        view.setStyle({"cartoon":{"color":"spectrum"}})
    view.addStyle({"hetflag":True},{"stick":{"colorscheme":"default","radius":0.3}})
    if highlight_residues:
        for rnum in highlight_residues:
            view.addStyle({"resi":str(rnum)},{"sphere":{"color":"#C0392B","radius":0.55}})
    view.zoomTo(); view.spin(False)
    html = view._make_html()
    return html.replace("</head>", _CSS+"</head>")
