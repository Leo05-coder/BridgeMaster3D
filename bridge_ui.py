import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from bridge_engine import run_bridge_optimization
import os
import struct
import time
import threading
import shutil

# Thread lock for Taichi access
ti_lock = threading.Lock()

def export_stl_final(data, filename, threshold=0.4):
    print(f"--- Turbo STL Export: {filename} ---")
    nelx, nely, nelz = data.shape
    voxels = data > threshold
    num_voxels = np.sum(voxels)
    if num_voxels == 0: return None
    
    faces = []
    # Vectorized check for faces - much faster than Python loop if possible, 
    # but let's optimize the existing loop with a faster structure
    for i, j, k in zip(*np.where(voxels)):
        v = lambda x,y,z: (float(i+x), float(j+y), float(k+z))
        if i==0 or not voxels[i-1,j,k]: 
            n=(-1,0,0); p1=v(0,0,0); p2=v(0,0,1); p3=v(0,1,1); p4=v(0,1,0)
            faces.extend([(n, p1, p2, p3), (n, p1, p3, p4)])
        if i==nelx-1 or not voxels[i+1,j,k]: 
            n=(1,0,0); p1=v(1,0,0); p2=v(1,1,0); p3=v(1,1,1); p4=v(1,0,1)
            faces.extend([(n, p1, p2, p3), (n, p1, p3, p4)])
        if j==0 or not voxels[i,j-1,k]: 
            n=(0,-1,0); p1=v(0,0,0); p2=v(1,0,0); p3=v(1,0,1); p4=v(0,0,1)
            faces.extend([(n, p1, p2, p3), (n, p1, p3, p4)])
        if j==nely-1 or not voxels[i,j+1,k]: 
            n=(0,1,0); p1=v(0,1,0); p2=v(0,1,1); p3=v(1,1,1); p4=v(1,1,0)
            faces.extend([(n, p1, p2, p3), (n, p1, p3, p4)])
        if k==0 or not voxels[i,j,k-1]: 
            n=(0,0,-1); p1=v(0,0,0); p2=v(0,1,0); p3=v(1,1,0); p4=v(1,0,0)
            faces.extend([(n, p1, p2, p3), (n, p1, p3, p4)])
        if k==nelz-1 or not voxels[i,j,k+1]: 
            n=(0,0,1); p1=v(0,0,1); p2=v(1,0,1); p3=v(1,1,1); p4=v(0,1,1)
            faces.extend([(n, p1, p2, p3), (n, p1, p3, p4)])

    with open(filename, 'wb') as f:
        f.write(b'\x00' * 80)
        f.write(struct.pack('<I', len(faces)))
        for n, p1, p2, p3 in faces:
            f.write(struct.pack('<3f3f3f3fH', *n, *p1, *p2, *p3, 0))
    return filename

def export_obj_final(data, filename, threshold=0.4):
    print(f"--- Turbo OBJ Export: {filename} ---")
    nelx, nely, nelz = data.shape
    voxels = data > threshold
    if not np.any(voxels): return None
    
    # Fast Vertex Mapping using NumPy
    # Vertices are the corners of the grid
    v_active = np.zeros((nelx+1, nely+1, nelz+1), dtype=np.int32)
    
    # Potential face vertices
    i, j, k = np.where(voxels)
    for di, dj, dk in np.ndindex(2, 2, 2):
        v_active[i+di, j+dj, k+dk] = 1
        
    v_indices = np.where(v_active)
    v_count = len(v_indices[0])
    v_map = np.zeros((nelx+1, nely+1, nelz+1), dtype=np.int32)
    v_map[v_indices] = np.arange(1, v_count + 1)
    
    # Build string content in memory
    out = ["# BridgeMaster3D OBJ Export\n"]
    for idx in range(v_count):
        out.append(f"v {v_indices[0][idx]} {v_indices[1][idx]} {v_indices[2][idx]}\n")
        
    # Face generation logic
    for ci, cj, ck in zip(*np.where(voxels)):
        p = lambda di, dj, dk: v_map[ci+di, cj+dj, ck+dk]
        # 6 faces
        if ci==0 or not voxels[ci-1,cj,ck]: out.append(f"f {p(0,0,0)} {p(0,0,1)} {p(0,1,1)} {p(0,1,0)}\n")
        if ci==nelx-1 or not voxels[ci+1,cj,ck]: out.append(f"f {p(1,0,0)} {p(1,1,0)} {p(1,1,1)} {p(1,0,1)}\n")
        if cj==0 or not voxels[ci,cj-1,ck]: out.append(f"f {p(0,0,0)} {p(1,0,0)} {p(1,0,1)} {p(0,0,1)}\n")
        if cj==nely-1 or not voxels[ci,cj+1,ck]: out.append(f"f {p(0,1,0)} {p(0,1,1)} {p(1,1,1)} {p(1,1,0)}\n")
        if ck==0 or not voxels[ci,cj,ck-1]: out.append(f"f {p(0,0,0)} {p(0,1,0)} {p(1,1,0)} {p(1,0,0)}\n")
        if ck==nelz-1 or not voxels[ci,cj,ck+1]: out.append(f"f {p(0,0,1)} {p(1,0,1)} {p(1,1,1)} {p(0,1,1)}\n")
        
    with open(filename, 'w') as f:
        f.write("".join(out))
    return filename

def apply_view(ax, mode):
    if mode == "Isometrisch":
        ax.view_init(elev=25, azim=-45)
    elif mode == "Oben (Draufsicht)":
        ax.view_init(elev=90, azim=-90)
    elif mode == "Seite (Längs)":
        ax.view_init(elev=0, azim=-90)
    elif mode == "Vorne (Querschnitt)":
        ax.view_init(elev=0, azim=0)

def generate_preview(bl, bw, bh, gl, gw, gh, curvature, a_pos, a_width, r_thick, lx_pct, lx_abs, l_type, view_mode, headroom):
    nx, ny, nz = int(gl), int(gw), int(gh)
    
    # Adaptive Downsampling for Performance
    # Downsample for preview performance (User complained about speed)
    # Target approx 30-40 voxels in max dim for fast responsiveness
    ds = max(1, int(max(nx, ny, nz) / 30))
    
    dnx, dny, dnz = nx // ds, ny // ds, nz // ds
    vx = np.zeros((dnx, dny, dnz))
    
    # Brückenhöhe (bh) ist nun direkt die Fahrbahnhöhe
    road_base_h = float(bh)
    
    for di in range(dnx):
        i = di * ds
        x_norm = (i + 0.5 * ds) / float(nx)
        para = 1.0 - 4.0 * (x_norm - 0.5)**2
        target_z = road_base_h + (curvature/100.0) * (float(bh) * 0.4) * para
        tk = int(target_z / ds)
        
        # Roadway (ID 2)
        for dt_idx in range(max(1, int(r_thick//ds))):
            kk = tk - dt_idx
            if 0 <= kk < dnz: vx[di, :, kk] = 2.0
        
        # Supports (ID 1)
        if ((i >= a_pos and i < a_pos + a_width) or (i >= nx - a_pos - a_width and i < nx - a_pos)):
            for dk in range(min(dnz, int(4/ds))):
                if dk < dnz:
                    # Only fill supports where nothing exists yet (preserve Road=2.0)
                    # Use masking to avoid overwriting existing road voxels
                    mask = (vx[di, :, dk] == 0)
                    vx[di, mask, dk] = 1.0

    bg_color = '#1a1a1a'
    text_color = '#cccccc'
    fig = plt.figure(figsize=(14, 10), facecolor=bg_color)
    ax = fig.add_subplot(111, projection='3d', facecolor=bg_color)
    ax.set_box_aspect((dnx, dny, dnz))
    apply_view(ax, view_mode)
    
    # Optimize Plotting: Single Call
    filled = vx > 0
    colors = np.empty(vx.shape, dtype=object)
    colors[vx == 1.0] = '#44cc6670' # Supports (Green, Alpha hex 70)
    colors[vx == 2.0] = '#4488ff70' # Road (Blue, Alpha hex 70)
    
    # Note: facecolors logic in matplotib handles alpha if embedded in hex or tuple
    # We use combined hex/alpha
    ax.voxels(filled, facecolors=colors)
    
    # Calculate height for load point visualization
    cx = lx_abs if lx_abs >= 0 else int((lx_pct/100.0) * float(nx))
    lx_norm_actual = (float(cx) + 0.5) / float(max(1, nx))
    lz = (road_base_h + (curvature/100.0) * (float(bh) * 0.4) * (1.0 - 4.0 * (lx_norm_actual - 0.5)**2)) / ds

    # EXCLUSION ZONE PREVIEW (Optimized: Fast Line instead of Voxel)
    # Always show if protect_road is True or if we just want to visualize the constraints
    ex_x = np.linspace(0, nx, 50)
    ex_x_norm = (ex_x + 0.5) / float(nx)
    ex_para = 1.0 - 4.0 * (ex_x_norm - 0.5)**2
    # Use bh (bridge_height) for roadmap height consistency
    ex_z = (road_base_h + (curvature/100.0) * (float(bh) * 0.4) * ex_para) / ds
    # Draw "Cage" for headroom (4 lines to form a 3D tunnel)
    y_left, y_right = 0, ny/ds
    # Bottom Left, Bottom Right, Top Left, Top Right of the road
    ax.plot(ex_x/ds, [y_left]*50, ex_z, color='#ff4444', linestyle='--', linewidth=1, alpha=0.3)
    ax.plot(ex_x/ds, [y_right]*50, ex_z, color='#ff4444', linestyle='--', linewidth=1, alpha=0.3)
    ax.plot(ex_x/ds, [y_left]*50, ex_z + (headroom/ds), color='#ff4444', linestyle='-', linewidth=1.5, alpha=0.6)
    ax.plot(ex_x/ds, [y_right]*50, ex_z + (headroom/ds), color='#ff4444', linestyle='-', linewidth=1.5, alpha=0.6)
    
    # Cross Connectors at start, middle, end
    for idx in [0, 25, 49]:
        xi, zi = ex_x[idx]/ds, ex_z[idx]
        zh = zi + headroom/ds
        ax.plot([xi, xi], [y_left, y_right], [zh, zh], color='#ff4444', linewidth=1.5, alpha=0.6)
        ax.plot([xi, xi], [y_left, y_left], [zi, zh], color='#ff4444', linewidth=1, alpha=0.3)
        ax.plot([xi, xi], [y_right, y_right], [zi, zh], color='#ff4444', linewidth=1, alpha=0.3)
    
    ny_ds = vx.shape[1]
    if l_type == "Punkt":
        # Visual point at EXACT center
        lz_pt = (road_base_h + (curvature/100.0) * (float(bh) * 0.4) * (1.0 - 4.0 * ((cx/nx) - 0.5)**2)) / ds
        ax.scatter([cx / ds], [ny / 2.0 / ds], [lz_pt + 2], color='#ff6666', s=300, label='Last')
    else:
        # Distributed Load: Arrows along the road
        # Plot 7 arrows distributed along the length
        for i_arrow in range(1, 8):
            lx_a = nx * (i_arrow / 8.0)
            ly_a = ny / 2.0
            x_norm_a = lx_a / float(nx)
            para_a = 1.0 - 4.0 * (x_norm_a - 0.5)**2
            lz_a = (road_base_h + (curvature/100.0) * (float(bh) * 0.4) * para_a) / ds
            
            # Quiver (arrow) pointing down
            # ax.quiver(x, y, z, u, v, w)
            ax.quiver(lx_a/ds, ly_a/ds, lz_a + 5, 0, 0, -4, color='red', linewidth=2, arrow_length_ratio=0.3)
        
    # Visually restore the scale on the axes even if downsampled
    ax.set_xticks(np.linspace(0, dnx, 5))
    ax.set_xticklabels([int(x*ds) for x in np.linspace(0, dnx, 5)], color=text_color)
    ax.set_yticks(np.linspace(0, dny, 5))
    ax.set_yticklabels([int(y*ds) for y in np.linspace(0, dny, 5)], color=text_color)
    ax.set_zticks(np.linspace(0, dnz, 5))
    ax.set_zticklabels([int(z*ds) for z in np.linspace(0, dnz, 5)], color=text_color)

    ax.set_xlabel("Länge (Voxel)", color=text_color)
    ax.set_ylabel("Breite (Voxel)", color=text_color)
    ax.set_zlabel("Höhe (Voxel)", color=text_color)
    ax.tick_params(colors=text_color)
    # Dark pane colors
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('#333333')
    ax.yaxis.pane.set_edgecolor('#333333')
    ax.zaxis.pane.set_edgecolor('#333333')

    plt.title("Vorschau | Blau=Brücke, Grün=Lager, Rot=Tunnel/Last", fontsize=14, color=text_color)
    # Entferne tight_layout (verursacht Warnungen bei 3D) und setze Ränder manuell
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9)
    # Explicitly clear and close figure to prevent memory warning
    res = (fig, "")
    plt.close(fig)
    return res

def simulate_bridge(bl, bw, bh, gl, gw, gh, efficiency, bend, a_pos, a_width, r_thick, lx_pct, lx_abs, l_type, rmin, threshold, protect_road, traffic_val, material_name, auto_calc, safety_factor, custom_path, headroom, progress=gr.Progress()):
    # Defensive checks for Gradio NoneType issues
    def safe_float(v, default):
        try: return float(v) if v is not None else default
        except: return default

    gl = safe_float(gl, 120)
    gw = safe_float(gw, 30)
    gh = safe_float(gh, 40)
    efficiency = safe_float(efficiency, 12)
    bend = safe_float(bend, 0)
    a_pos = safe_float(a_pos, 5)
    a_width = safe_float(a_width, 4)
    r_thick = safe_float(r_thick, 2)
    lx_pct = safe_float(lx_pct, 50)
    lx_abs = safe_float(lx_abs, -1)
    traffic_val = safe_float(traffic_val, 300)
    bh = safe_float(bh, 30)
    bl = safe_float(bl, 60)
    bw = safe_float(bw, 30)
    
    # Kopffreiheit ist nun immer aktiv (protect_road=True)
    protect_road = True
    
    if material_name is None: material_name = "Stahl"
    
    # feedback to user immediately
    gr.Info("Simulation gestartet... bitte warten.")
    # If auto-calculate is on, start with a slightly higher volume to ensure initial connection in large blocks
    init_vol = 0.15 if auto_calc else float(efficiency) / 100.0
    volfrac = init_vol
    curvature = float(bend) / 100.0
    
    try:
        from bridge_engine import MATERIALS # Import here to ensure we get updated dictionary
        with ti_lock:
            final_data = run_bridge_optimization(
                int(gl), int(gw), int(gh), 
                volfrac, rmin, curvature, 
                anchor_pos=a_pos, anchor_width=a_width, road_thick=r_thick,
                load_x_pct=(lx_pct/100.0), load_x_abs=lx_abs, load_type=l_type,
                protect_road=(1 if protect_road else 0),
                traffic_load=float(traffic_val),
                material_name=material_name,
                auto_calc=auto_calc,
                safety_factor=safety_factor,
                bridge_height=bh,
                headroom=headroom,
                progress_callback=lambda s, t: progress(s/t, desc=f"Schritt {s}/{t}")
            )
        
        # Calculate final stats
        vox_vol = np.sum(final_data > threshold)
        total_p = final_data.size
        actual_vol_pct = (vox_vol / total_p) * 100.0
        
        mat = MATERIALS.get(material_name, MATERIALS["Stahl"])
        # Mass Calc: Assume 5x5x5mm voxels for PLA context (user request)
        # 1 voxel = 0.5cm * 0.5cm * 0.5cm = 0.125 cm^3
        # mass = count * 0.125 * rho
        vol_cm3_per_voxel = 0.125
        mass_kg = vox_vol * vol_cm3_per_voxel * mat["rho"] / 1000.0
        
        # Naming Logic: Nummer"Bridge"HöhexBreitexLänge
        counter_file = "counter.txt"
        count = 1
        if os.path.exists(counter_file):
            with open(counter_file, "r") as f:
                try: count = int(f.read().strip()) + 1
                except: count = 1
        with open(counter_file, "w") as f:
            f.write(str(count))
            
        filename_stl = f"{count}Bridge_{int(bl)}x{int(bw)}x{int(bh)}.stl"
        filename_obj = f"{count}Bridge_{int(bl)}x{int(bw)}x{int(bh)}.obj"
        
        # Use a temp folder for exports (auto-cleaned, not the user's Output folder)
        temp_dir = os.path.join(os.getcwd(), ".temp_export")
        if os.path.exists(temp_dir):
            # Clean old temp files
            for old_f in os.listdir(temp_dir):
                old_path = os.path.join(temp_dir, old_f)
                if os.path.isfile(old_path) and (old_f.endswith('.stl') or old_f.endswith('.obj')):
                    try: os.remove(old_path)
                    except: pass
        os.makedirs(temp_dir, exist_ok=True)
        
        stl_path = os.path.join(temp_dir, filename_stl)
        obj_path = os.path.join(temp_dir, filename_obj)
        
        # Remember the user's desired output dir for manual download
        user_output_dir = custom_path if custom_path and custom_path.strip() else os.path.join(os.getcwd(), "Output")
        
        res_stl = export_stl_final(final_data, stl_path, threshold=threshold) 
        res_obj = export_obj_final(final_data, obj_path, threshold=threshold)
        
        if res_stl is None:
            gr.Warning("Statik-Fehler: Keine Last tragende Verbindung gefunden!")
            return None, None, gr.update(selected=0), "Fehler"
            
        mass_str = f"{mass_kg:.3f}".replace('.', ',')
        rho_val = mat["rho"]
        stats_msg = f"### ✅ Ergebnis\n- **Material:** {material_name} ({rho_val} g/cm³)\n- **Verbrauch:** {actual_vol_pct:.1f}% des Bauraums\n- **Masse:** {mass_str} kg\n- **Lasteinleitung:** {l_type}"
        if auto_calc:
            stats_msg = "### 🤖 Auto-Mengenberechnung Errechnet:\n" + stats_msg
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        gr.Error(f"Simulation abgestürzt: {str(e)}")
        return None, None, gr.update(selected=0), f"Fehler: {str(e)}"

    # Try relative path for Model3D to improve loading on some browsers
    rel_stl = os.path.relpath(stl_path) if stl_path.startswith(os.getcwd()) else stl_path
    # Store file paths + output dir in state (no auto-download)
    return rel_stl, {"files": [stl_path, obj_path], "output_dir": user_output_dir}, gr.update(selected=1), stats_msg

def sync_l(val): return int(val * 1.10)
def sync_w(val): return int(val * 1.10)
def sync_headroom(bh_val): return max(3, int(bh_val * 0.3))
def sync_h(bh_val, head_val): 
    # Formel: bh + headroom + 10% bh = gh
    return int(bh_val * 1.10 + head_val)

def mark_dirty():
    return "⚠️ Änderungen vorgenommen. Bitte Vorschau aktualisieren."

def handle_abs_change(val):
    return gr.update(interactive=(val < 0))

def suggest_rmin(length):
    suggested = max(1.5, length / 40.0 * 1.2)
    return round(suggested, 1)

with gr.Blocks(title="BridgeMaster3D") as demo:
    gr.Markdown("# 🏗️ BridgeMaster3D: Dein Brücken-Designer", elem_id="main_title")
    
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group(elem_id="bridge_dims_group"):
                gr.Markdown("### 📐 Brückenmaße (Voxel)")
                bl_slider = gr.Slider(1, 240, value=60, step=1, label="Brücken-Länge (Spannweite)", elem_id="bl_slider")
                bw_slider = gr.Slider(1, 80, value=20, step=1, label="Brücken-Breite (Fahrbahn)", elem_id="bw_slider")
                bh_slider = gr.Slider(1, 100, value=15, step=1, label="Brücken-Höhe (Scheitel)", elem_id="bh_slider")
                
                with gr.Accordion("Expert: Design-Raum (Blockmaße)", open=False, elem_id="acc_block"):
                    gr.Markdown("*Hier kannst du den Block vergrößern, um der KI mehr 'Luft' für Stützen zu geben.*")
                    gl_slider = gr.Slider(1, 240, value=66, step=1, label="Block-Länge (X)", elem_id="id_gl")
                    gw_slider = gr.Slider(1, 80, value=22, step=1, label="Block-Breite (Y)", elem_id="id_gw")
                    gh_slider = gr.Slider(1, 100, value=22, step=1, label="Block-Höhe (Z)", elem_id="id_gh")
            
            with gr.Group(elem_id="roadway_group_container"):
                gr.Markdown("### 🎢 Straßenverlauf")
                bend_slider = gr.Slider(-50, 50, value=0, step=1, label="Biegung (%)", elem_id="id_bend")
                thick_slider = gr.Slider(1, 8, value=2, step=1, label="Stärke der Fahrbahn", elem_id="id_thick")
                headroom_slider = gr.Slider(0, 50, value=5, step=1, label="🛡️ Kopffreiheit (Höhe in Voxel)", elem_id="id_headroom")
                # Checkbox wurde gemäß Benutzerwunsch entfernt (Sperre immer aktiv wenn > 0)
                
            with gr.Accordion("🎯 Last (Kraft)", open=False, elem_id="load_group_container"):
                l_type = gr.Radio(["Punkt", "Fläche (Verteilt)"], value="Punkt", label="Last-Typ", elem_id="id_ltype")
                traffic_force_slider = gr.Slider(minimum=0, maximum=1000, value=0, step=1, label="Gewicht in Kg", elem_id="id_force")
                lx_abs = gr.Slider(-1, 240, value=-1, step=1, label="Feste X-Position (-1 = Deaktiviert)", elem_id="id_lxabs")
                lx_pct = gr.Slider(0, 100, value=50, step=1, label="Last-Position X (%)", elem_id="id_lxpct")
                
            with gr.Accordion("⚓ Verankerungen", open=False, elem_id="anchor_group_container"):
                pos_slider = gr.Slider(0, 100, value=5, step=1, label="Abstand vom Rand", elem_id="id_pos")
                width_slider = gr.Slider(1, 40, value=4, step=1, label="Breite der Pfeiler", elem_id="id_width")
            
            with gr.Accordion("🛠️ Erweiterte Simulationseinstellungen", open=False, elem_id="acc_expert"):
                rmin_slider = gr.Slider(1.0, 10.0, value=1.5, step=0.1, label="Filter-Radius (Strahldicke)")
                gr.Markdown("**Filter-Radius:** Bestimmt die Mindestdicke der Träger. Werte ab 1.5 verhindern schwache Eck-Verbindungen. **Symmetrie:** Die Brücke wird automatisch symmetrisch (Längs & Quer) optimiert.")
                
                threshold_slider = gr.Slider(0.1, 0.9, value=0.4, step=0.05, label="STL-Schwellenwert")
                gr.Markdown("**Schwellenwert:** Bestimmt, ab welcher Dichte Material als 'fest' exportiert wird.")
                
            with gr.Group(elem_id="ki_opt_group_container"):
                gr.Markdown("### 🏗️ KI-Ziele")
                auto_check = gr.Checkbox(True, label="🤖 Materialbedarf automatisch berechnen (Norm)")
                
                safety_slider = gr.Slider(1.0, 5.0, value=1.5, step=0.1, label="Sicherheitsfaktor", visible=True)
                safety_info = gr.Markdown("*Die KI sucht die minimale Menge, um diesen Faktor zu erfüllen.*", visible=True)

                eff_slider = gr.Slider(5, 60, value=12, step=1, label="Manuelle Materialmenge (Vol %)", visible=False, interactive=False, elem_id="eff_slider")
            
        with gr.Column(scale=3):
            with gr.Tabs(elem_id="main_tabs") as tabs:
                with gr.TabItem("1. Entwurf (Vorschau)", id=0, elem_id="tab_sketch"):
                    preview_plot = gr.Plot(elem_id="preview_canvas")
                    with gr.Row():
                        with gr.Column(scale=2):
                            view_mode = gr.Dropdown(["Isometrisch", "Oben (Draufsicht)", "Seite (Längs)", "Vorne (Querschnitt)"], value="Isometrisch", label="Perspektive", elem_id="view_mode")
                            mat_dd = gr.Dropdown(["Stahl", "Beton", "Holz", "Carbon", "PLA (3D-Druck)"], value="PLA (3D-Druck)", label="Material")
                        with gr.Column(scale=1):
                            update_preview_btn = gr.Button("🔄 Vorschau erneuern", variant="secondary", elem_id="refresh_btn")
                            run_btn = gr.Button("🚀 Optimierung Starten", variant="primary", elem_id="run_btn")
                    dirty_label = gr.Markdown("", elem_id="dirty_label")
                    
                with gr.TabItem("2. Output", id=1, elem_id="tab_output"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            result_plot = gr.Model3D(label="Interaktive 3D Brücke", clear_color=[0.1, 0.1, 0.1, 0], elem_id="id_3d_model")
                        with gr.Column(scale=1):
                            res_stats = gr.Markdown("### 📊 Stats\nNoch keine Simulation durchgeführt.", elem_id="res_stats")
                    
                    gr.Markdown("*Das Modell kann nach Abschluss im Browser frei gedreht werden.*")
                    path_input = gr.Textbox(label="Lokaler Speicherpfad (Archiv)", value=os.path.join(os.getcwd(), "Output"), elem_id="id_path")
                    last_files = gr.State(value=None)
                    download_btn = gr.Button("📥 Modelle herunterladen (STL & OBJ)", variant="secondary", elem_id="download_btn")
                    stl_out = gr.File(label="Dateien", file_count="multiple", elem_id="id_stl_download", visible=False)

    # Interactivity for Auto-Calc
    def toggle_auto(val):
        return gr.update(visible=val), gr.update(visible=val), gr.update(visible=not val, interactive=not val)
    
    auto_check.change(toggle_auto, auto_check, [safety_slider, safety_info, eff_slider])

    bl_slider.change(sync_l, bl_slider, gl_slider)
    bw_slider.change(sync_w, bw_slider, gw_slider)
    
    # Sync gh based on bh AND headroom
    bh_slider.change(sync_h, [bh_slider, headroom_slider], gh_slider)
    bh_slider.change(sync_headroom, bh_slider, headroom_slider)
    headroom_slider.change(sync_h, [bh_slider, headroom_slider], gh_slider)
    
    bl_slider.change(suggest_rmin, bl_slider, rmin_slider)
    lx_abs.change(handle_abs_change, lx_abs, lx_pct)
    
    # Triggers for Dirty Label
    preview_trigger_inputs = [bl_slider, bw_slider, bh_slider, bend_slider, pos_slider, width_slider, thick_slider, lx_pct, lx_abs, headroom_slider, l_type]
    for inp in preview_trigger_inputs:
        if hasattr(inp, 'change'):
            inp.change(mark_dirty, None, dirty_label)

    preview_inputs = [bl_slider, bw_slider, bh_slider, gl_slider, gw_slider, gh_slider, bend_slider, pos_slider, width_slider, thick_slider, lx_pct, lx_abs, l_type, view_mode, headroom_slider]
    
    # MANUAL PREVIEW: Remove automatic triggers from sliders to avoid race conditions.
    # The user now uses the "Vorschau erneuern" button explicitly.
    update_preview_btn.click(generate_preview, preview_inputs, [preview_plot, dirty_label])
    view_mode.change(generate_preview, preview_inputs, [preview_plot, dirty_label])
    
    # Updated sim_inputs to include bl, bw for filenames
    sim_inputs = [bl_slider, bw_slider, bh_slider, gl_slider, gw_slider, gh_slider, eff_slider, bend_slider, pos_slider, width_slider, thick_slider, lx_pct, lx_abs, l_type, rmin_slider, threshold_slider, gr.State(True), traffic_force_slider, mat_dd, auto_check, safety_slider, path_input, headroom_slider]
    
    def start_opt_trigger():
        # This function only switches the tab initially
        return gr.update(selected=1)

    def on_sim_done(result_stl, file_list, tab_update, stats):
        # Store file paths in state, don't push to File widget yet
        return result_stl, file_list, tab_update, stats

    # Multi-Action Button: Switch tab AND run simulation
    run_btn.click(start_opt_trigger, None, tabs).then(
        simulate_bridge, sim_inputs, [result_plot, last_files, tabs, res_stats]
    )
    
    # Manual download: copy files to Output folder and offer download
    def trigger_download(state_data, output_path):
        if not state_data or "files" not in state_data:
            gr.Warning("Noch keine Simulation durchgeführt.")
            return gr.update(visible=False)
        
        files = state_data["files"]
        # Copy to user's chosen Output folder
        save_dir = output_path if output_path and output_path.strip() else state_data.get("output_dir", os.path.join(os.getcwd(), "Output"))
        os.makedirs(save_dir, exist_ok=True)
        
        saved_files = []
        for fp in files:
            if os.path.exists(fp):
                dest = os.path.join(save_dir, os.path.basename(fp))
                shutil.copy2(fp, dest)
                saved_files.append(dest)
        
        if saved_files:
            gr.Info(f"Dateien gespeichert in: {save_dir}")
            return gr.update(value=saved_files, visible=True)
        gr.Warning("Fehler beim Kopieren der Dateien.")
        return gr.update(visible=False)
    
    download_btn.click(trigger_download, [last_files, path_input], stl_out)
    
    demo.load(generate_preview, preview_inputs, [preview_plot, dirty_label])

if __name__ == "__main__":
    demo.queue().launch(theme=gr.themes.Soft(), share=False)
