import taichi as ti
import numpy as np
import time
from collections import deque
import struct

# Taichi INITIALIZATION (Fixed for life of the process)
# We MUST initialize on the main thread to avoid std::thread assertion failures
print("Initialisiere Taichi-Motor (Main Thread)...")
try:
    ti.init(arch=ti.cuda, device_memory_GB=2.0)
except Exception as e:
    print(f"CUDA Init failed: {e}. Falling back to CPU.")
    ti.init(arch=ti.cpu)

# DUMMY KERNEL to force SNode materialization on the main thread
@ti.kernel
def warmup():
    x[0, 0, 0] = 0.0

# MAXIMUM DIMENSIONS (Fixed Werkstatt)
MX, MY, MZ = 240, 80, 100

# GLOBAL FIELDS (Allocated once)
x = ti.field(ti.f32, shape=(MX, MY, MZ))
xPhys = ti.field(ti.f32, shape=(MX, MY, MZ))
u = ti.Vector.field(3, ti.f32, shape=(MX+1, MY+1, MZ+1))
f = ti.Vector.field(3, ti.f32, shape=(MX+1, MY+1, MZ+1))
f_ext = ti.Vector.field(3, ti.f32, shape=(MX+1, MY+1, MZ+1))
dc = ti.field(ti.f32, shape=(MX, MY, MZ))

# Solver & Support Fields
r, p, Ap = [ti.Vector.field(3, ti.f32, shape=(MX+1, MY+1, MZ+1)) for _ in range(3)]
fixed = ti.field(ti.i32, shape=(MX+1, MY+1, MZ+1))
passive = ti.field(ti.i32, shape=(MX, MY, MZ))
forbidden = ti.field(ti.i32, shape=(MX, MY, MZ))

# MATERIALS (Properties: E [GPa], rho [g/cm³], sigma_y [MPa])
MATERIALS = {
    "Stahl": {"E": 210.0, "rho": 7.85, "sigma_y": 235.0},
    "Beton": {"E": 30.0, "rho": 2.4, "sigma_y": 25.0},
    "Holz": {"E": 12.0, "rho": 0.5, "sigma_y": 40.0},
    "Carbon": {"E": 150.0, "rho": 1.6, "sigma_y": 1000.0},
    "PLA (3D-Druck)": {"E": 3.5, "rho": 1.25, "sigma_y": 50.0}
}

E0, Emin = 1.0, 1e-6
# Global stress field
vonmises = ti.field(ti.f32, shape=(MX, MY, MZ))

# ================= KERNELS (Module Level) =================

@ti.kernel
def reset_all_fields(nx: ti.i32, ny: ti.i32, nz: ti.i32):
    for i, j, k in ti.ndrange(nx + 1, ny + 1, nz + 1):
        u[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
        f[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
        f_ext[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
        fixed[i, j, k] = 0
        r[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
        p[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
        Ap[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
    for i, j, k in ti.ndrange(nx, ny, nz):
        x[i, j, k] = 0.0
        xPhys[i, j, k] = 0.0
        dc[i, j, k] = 0.0
        vonmises[i, j, k] = 0.0
        passive[i, j, k] = 0
        forbidden[i, j, k] = 0

@ti.kernel
def init_setup_kernel(nx: ti.i32, ny: ti.i32, nz: ti.i32, curv: ti.f32, a_pos: ti.f32, a_width: ti.f32, r_thick: ti.f32, lx_pct: ti.f32, lx_abs: ti.f32, l_type: ti.i32, vol: ti.f32, protect: ti.i32, f_mag: ti.f32, road_base_h: ti.f32, headroom: ti.f32):
    base_h = road_base_h
    for i, j, k in ti.ndrange(nx, ny, nz):
        x[i, j, k] = vol # Starting density
        # ROADWAY CALCULATION
        x_norm = 0.5
        if nx > 1: x_norm = (float(i) + 0.5) / float(nx)
        para = 1.0 - 4.0 * (x_norm - 0.5)**2
        # Fix: Nutze Brückenhöhe (base_h) direkt als Fahrbahnhöhe
        # Der Faktor 0.4 passt zur Vorschau (dort float(bh) * 0.4)
        target_z = base_h + curv * (float(base_h) * 0.4) * para
        tk = int(target_z)
        
        # ROADWAY (Passive)
        if k >= tk - (int(r_thick)-1) and k <= tk:
            passive[i, j, k] = 1; x[i, j, k] = 1.0
            
        # EXCLUSION ZONE (Sperrzone über Fahrbahn)
        if protect == 1 and k > tk and k <= tk + int(headroom):
            forbidden[i, j, k] = 1; x[i, j, k] = 0.0

        # SUPPORTS
        if ((i >= a_pos and i < a_pos + a_width) or (i >= nx - a_pos - a_width and i < nx - a_pos)) and k < 3:
            passive[i, j, k] = 1; x[i, j, k] = 1.0
            if k == 0:
                for di, dj in ti.ndrange(2, 2): fixed[i+di, j+dj, 0] = 1

    # LOAD (Precision Splitting)
    # Target X
    cx = lx_abs if lx_abs >= 0 else lx_pct * float(nx)
    # Target Y (Middle of road)
    cy = float(ny) / 2.0
    
    # 3D Position tracking for Load
    cx_idx = int(cx)
    cy_idx = int(cy)
    
    # Parabolic height at load position
    lx_norm = (float(cx) + 0.5) / float(max(1, nx))
    # Fix: Auch hier Brückenhöhe (base_h) und Faktor 0.4 nutzen
    l_tk = int(base_h + curv * (float(base_h) * 0.4) * (1.0 - 4.0 * (lx_norm - 0.5)**2))
    fz = min(nz, l_tk + 1)
    
    if l_type == 0: # Point Load
        # Split across 4 nodes in the top face of the voxel if needed, 
        # but here we focus on Y-centering per user request.
        y_frac = cy - float(cy_idx)
        # Distribute on the center line nodes
        # Node (cx, cy_idx, fz) and (cx, cy_idx+1, fz)
        f_ext[int(cx), cy_idx, fz].z += f_mag * (1.0 - y_frac)
        f_ext[int(cx), cy_idx + 1, fz].z += f_mag * y_frac
    else: # Line Load
        for jj in range(ny + 1): f_ext[int(cx), jj, fz].z += f_mag / float(ny+1)

@ti.kernel
def apply_gravity_kernel(nx: ti.i32, ny: ti.i32, nz: ti.i32, gravity: ti.f32):
    for i, j, k in ti.ndrange(nx + 1, ny + 1, nz + 1): f[i, j, k] = f_ext[i, j, k]
    for i, j, k in ti.ndrange(nx, ny, nz):
        if xPhys[i, j, k] > 0.01:
            force = -xPhys[i, j, k] * gravity / 8.0
            for di, dj, dk in ti.ndrange(2, 2, 2): f[i+di, j+dj, k+dk].z += force

@ti.kernel
def apply_ku_kernel(nx: ti.i32, ny: ti.i32, nz: ti.i32, v_field: ti.template(), p_val: ti.f32):
    for i, j, k in ti.ndrange(nx + 1, ny + 1, nz + 1): Ap[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
    for i, j, k in ti.ndrange(nx, ny, nz):
        k_el = Emin + xPhys[i, j, k]**p_val * (E0 - Emin)
        for di, dj, dk in ti.ndrange(2, 2, 2):
            for oi, oj, ok in ti.ndrange(2, 2, 2):
                if not (di == oi and dj == oj and dk == ok):
                    edge = ti.Vector([float(di-oi), float(dj-oj), float(dk-ok)])
                    rel_v = v_field[i+di, j+dj, k+dk] - v_field[i+oi, j+oj, k+ok]
                    Ap[i+di, j+dj, k+dk] += edge * (rel_v.dot(edge)) * k_el / (edge.norm_sqr()**2 + 1e-10)

@ti.kernel
def cg_init_kernel(nx: ti.i32, ny: ti.i32, nz: ti.i32) -> ti.f32:
    rho = 0.0
    for i, j, k in ti.ndrange(nx + 1, ny + 1, nz + 1):
        if fixed[i, j, k] == 0:
            r[i, j, k] = f[i, j, k] - Ap[i, j, k]
            p[i, j, k] = r[i, j, k]
            rho += r[i, j, k].dot(r[i, j, k])
    return rho

@ti.kernel
def cg_step_pAp_kernel(nx: ti.i32, ny: ti.i32, nz: ti.i32) -> ti.f32:
    pAp = 0.0
    for i, j, k in ti.ndrange(nx + 1, ny + 1, nz + 1):
        if fixed[i, j, k] == 0: pAp += p[i, j, k].dot(Ap[i, j, k])
    return pAp

@ti.kernel
def cg_step_ur_kernel(nx: ti.i32, ny: ti.i32, nz: ti.i32, alpha: ti.f32) -> ti.f32:
    rho_new = 0.0
    for i, j, k in ti.ndrange(nx + 1, ny + 1, nz + 1):
        if fixed[i, j, k] == 0:
            u[i, j, k] += alpha * p[i, j, k]
            r[i, j, k] -= alpha * Ap[i, j, k]
            rho_new += r[i, j, k].dot(r[i, j, k])
    return rho_new

@ti.kernel
def cg_step_p_kernel(nx: ti.i32, ny: ti.i32, nz: ti.i32, beta: ti.f32):
    for i, j, k in ti.ndrange(nx + 1, ny + 1, nz + 1):
        if fixed[i, j, k] == 0: p[i, j, k] = r[i, j, k] + beta * p[i, j, k]

@ti.kernel
def filter_vars_kernel(nx: ti.i32, ny: ti.i32, nz: ti.i32, f_radius: ti.f32):
    ir = int(f_radius)
    for i, j, k in ti.ndrange(nx, ny, nz):
        if passive[i, j, k] == 1:
            xPhys[i, j, k] = 1.0
        elif forbidden[i, j, k] == 1:
            xPhys[i, j, k] = 0.0
        else:
            tw, txw = 0.0, 0.0
            for di, dj, dk in ti.ndrange((-ir, ir+1), (-ir, ir+1), (-ir, ir+1)):
                ni, nj, nk = i+di, j+dj, k+dk
                if 0 <= ni < nx and 0 <= nj < ny and 0 <= nk < nz:
                    w = max(0.0, f_radius - ti.sqrt(float(di**2 + dj**2 + dk**2)))
                    tw += w; txw += w * x[ni, nj, nk]
            xPhys[i, j, k] = txw / (tw + 1e-10)

@ti.kernel
def calc_sensitivity_kernel(nx: ti.i32, ny: ti.i32, nz: ti.i32, p_val: ti.f32):
    for i, j, k in ti.ndrange(nx, ny, nz):
        k_g = p_val * (xPhys[i, j, k]**(p_val-1)) * (E0 - Emin)
        energy = 0.0
        for di, dj, dk in ti.ndrange(2, 2, 2):
            for oi, oj, ok in ti.ndrange(2, 2, 2):
                if not (di == oi and dj == oj and dk == ok):
                    edge = ti.Vector([float(di-oi), float(dj-oj), float(dk-ok)])
                    rel_u = u[i+di, j+dj, k+dk] - u[i+oi, j+oj, k+ok]
                    energy += (rel_u.dot(edge))**2 / (edge.norm_sqr() + 1e-10)
        dc[i, j, k] = -k_g * energy
        if passive[i, j, k] == 1: dc[i, j, k] = 0.0

@ti.kernel
def calc_stress_kernel(nx: ti.i32, ny: ti.i32, nz: ti.i32):
    # Stress estimation from Strain Energy
    # Sigma is independent of absolute E for fixed external loads.
    # We use a heuristic scaling to match MPa range for typical bridge cases.
    for i, j, k in ti.ndrange(nx, ny, nz):
        energy = 0.0
        if xPhys[i, j, k] > 0.05:
            # Local energy density
            for di, dj, dk in ti.ndrange(2, 2, 2):
                for oi, oj, ok in ti.ndrange(2, 2, 2):
                    if not (di == oi and dj == oj and dk == ok):
                        edge = ti.Vector([float(di-oi), float(dj-oj), float(dk-ok)])
                        rel_u = u[i+di, j+dj, k+dk] - u[i+oi, j+oj, k+ok]
                        energy += (rel_u.dot(edge))**2 / (edge.norm_sqr() + 1e-10)
            
            # Calibration: energy ~ epsilon^2. Sigma ~ E * epsilon ~ sqrt(E_eff * energy).
            # But since we solved with E_engine = 1.0, u is u_normalized.
            # Sigma_approx = sqrt(energy) * factor
            # With f_mag scaling, factor=40.0 yields realistic MPa results (range 5-200)
            vonmises[i, j, k] = ti.sqrt(max(0.0, energy)) * 40.0 
        else:
            vonmises[i, j, k] = 0.0

@ti.kernel
def symmetry_dc_kernel(nx: ti.i32, ny: ti.i32, nz: ti.i32):
    # X-Symmetry (Length) and Y-Symmetry (Width)
    # Since the road and load are centered in Y, Y-symmetry is physically correct.
    for i, j, k in ti.ndrange(nx, ny, nz):
        # Mirror indices
        mi = nx - 1 - i
        mj = ny - 1 - j
        
        # Average across 4 symmetric points: (i,j), (mi,j), (i,mj), (mi,mj)
        v = (dc[i, j, k] + dc[mi, j, k] + dc[i, mj, k] + dc[mi, mj, k]) / 4.0
        dc[i, j, k] = v

@ti.kernel
def enforce_passive_kernel(nx: ti.i32, ny: ti.i32, nz: ti.i32):
    for i, j, k in ti.ndrange(nx, ny, nz):
        if passive[i, j, k] == 1: x[i, j, k] = 1.0
        if forbidden[i, j, k] == 1: x[i, j, k] = 0.0

@ti.kernel
def oc_update_kernel(nx: ti.i32, ny: ti.i32, nz: ti.i32, lmid: ti.f32, move: ti.f32) -> ti.f32:
    total_vol = 0.0
    for i, j, k in ti.ndrange(nx, ny, nz):
        x_old = x[i, j, k]
        # xnew = max(0, max(x-move, min(1, min(x+move, x * sqrt(-dc/lambda)))))
        x_new = ti.max(0.0, ti.max(x_old - move, ti.min(1.0, ti.min(x_old + move, x_old * ti.sqrt(ti.max(0.0, -dc[i, j, k] / (lmid + 1e-10)))))))
        x[i, j, k] = x_new
        total_vol += x_new
    return total_vol

# ================= WRAPPER FUNCTION =================

def run_bridge_optimization(nelx, nely, nelz, volfrac, rmin, curvature, anchor_pos=8.0, anchor_width=8.0, road_thick=2.0, load_x_pct=0.5, load_x_abs=-1.0, load_type="Punkt", protect_road=1, traffic_load=300.0, material_name="Stahl", auto_calc=False, safety_factor=1.5, bridge_height=None, headroom=10.0, progress_callback=None):
    # Sanity check bounds
    nx, ny, nz = min(nelx, MX), min(nely, MY), min(nelz, MZ)
    
    # Get Material Props
    mat = MATERIALS.get(material_name, MATERIALS["Stahl"])
    E_mat = mat["E"]
    rho_mat = mat["rho"]
    sigma_y = mat["sigma_y"]
    
    # Target Stress (including safety factor)
    target_sigma = sigma_y / max(1.0, safety_factor)
    
    # Base height for road (intended bridge height center)
    bh = bridge_height if bridge_height else nelz
    
    # Startvolumen für Automatik auf 20% erhöhen (Stabilität), sonst User-Wert
    current_vol = 0.20 if auto_calc else volfrac
    max_outer_loops = 3 if auto_calc else 1
    # Pass 1 (Automatik): 60 Schritte für stabilere Volumen-Schätzung (vorher 40). 
    # Pass 2: 60, Pass 3: 100.
    total_expected_iters = (60 if auto_calc else 100) + (max_outer_loops - 1) * 80 if auto_calc else 100
    
    iters_done = 0
    for outer in range(max_outer_loops):
        # Erste Durchläufe etwas länger für Symmetrie-Findung, letzter Durchlauf fein.
        inner_loops = 60 if (auto_calc and outer < max_outer_loops-1) else 100
        
        # Reset WORKSHOP for each design pass (to avoid bias from previous volume)
        reset_all_fields(nx, ny, nz)
        
        # 1. INITIAL SETUP
        l_type_ti = 0 if load_type == "Punkt" else 1
        # f_mag calibration: 300kg -> ~3.0. Realistic for Taichi solver range.
        f_mag = -max(0.1, abs(traffic_load) / 100.0)
        
        # Brückenhöhe (bh) ist nun direkt die Fahrbahnhöhe
        road_base_h = float(bh)
        init_setup_kernel(nx, ny, nz, curvature, anchor_pos, anchor_width, road_thick, load_x_pct, load_x_abs, l_type_ti, current_vol, protect_road, f_mag, road_base_h, float(headroom))
        enforce_passive_kernel(nx, ny, nz)
        
        penal = 3.0
        for loop in range(1, inner_loops + 1):
            current_move = 0.3 - (0.25 * loop / inner_loops)
            
            # Step A: Filter
            filter_vars_kernel(nx, ny, nz, rmin)
            
            # Step B: Finite Element Analysis (FEA)
            # Self-weight: density * g. Scaled to taichi range.
            apply_gravity_kernel(nx, ny, nz, 0.002 * rho_mat) 
            
            apply_ku_kernel(nx, ny, nz, u, penal)
            rho = cg_init_kernel(nx, ny, nz)
            # Speedup: In den ersten Schritten der Volumenfindung reichen weniger Iterationen
            max_cg_iters = 400 if (not auto_calc or loop > 10) else 200
            for _ in range(max_cg_iters):
                if rho < 1e-12: break
                apply_ku_kernel(nx, ny, nz, p, penal)
                pAp = cg_step_pAp_kernel(nx, ny, nz)
                alpha = rho / (pAp + 1e-18)
                rho_new = cg_step_ur_kernel(nx, ny, nz, alpha)
                cg_step_p_kernel(nx, ny, nz, rho_new / (rho + 1e-16))
                rho = rho_new
                
            # Step C: Sensitivities
            calc_sensitivity_kernel(nx, ny, nz, penal)
            
            if nx > 1:
                is_centered = abs(load_x_pct - 0.5) < 0.02 and load_x_abs < 0
                if is_centered:
                    symmetry_dc_kernel(nx, ny, nz)
            
            # Step D: Optimality Criteria
            dc_np = dc.to_numpy()[:nx, :ny, :nz]
            x_np = x.to_numpy()[:nx, :ny, :nz]
            
            l1, l2 = 0, 1e9
            for _ in range(64):
                lmid = 0.5*(l2+l1)
                xnew = np.maximum(0,np.maximum(x_np-current_move,np.minimum(1,np.minimum(x_np+current_move, x_np*np.sqrt(np.maximum(0.0,-dc_np/(lmid+1e-10)))))))
                if xnew.sum() > current_vol*nx*ny*nz: l1 = lmid
                else: l2 = lmid
            
            x_full = np.zeros((MX, MY, MZ), dtype=np.float32)
            x_full[:nx, :ny, :nz] = xnew
            x.from_numpy(x_full)
            enforce_passive_kernel(nx, ny, nz)
            
            if progress_callback and loop % 2 == 0:
                # Adjust progress for outer loops
                prog_val = (iters_done + loop) / total_expected_iters
                progress_callback(min(99, int(prog_val * 100)), 100)

        iters_done += inner_loops

        # After each pass: Calculate Stress
        calc_stress_kernel(nx, ny, nz)
        vm_np = vonmises.to_numpy()[:nx, :ny, :nz]
        # Ignore outliers (97.0th percentile) to allow removing even more material
        max_vm = np.percentile(vm_np[xPhys.to_numpy()[:nx,:ny,:nz] > 0.4], 97.0) if np.any(xPhys.to_numpy()[:nx,:ny,:nz] > 0.4) else 0.0
        
        if not auto_calc:
            break
        
        # Auto-Calc Volume Adjustment
        # SF = sigma_y / max_vm. We want SF = safety_factor.
        # Required volume scale = max_vm / target_sigma
        # Auto-Calc Volume Adjustment with Damping (0.5) to prevent oscillations
        vol_scale = max_vm / (target_sigma + 1e-9)
        # Apply square root damping for volume scaling (linear stress vs quadratic energy approx)
        vol_scale = ti.math.sqrt(vol_scale) if vol_scale > 1.0 else vol_scale 
        
        # Limit the step size for stability (Lower limit from 0.7 to 0.3 for faster reduction)
        vol_scale = max(0.3, min(1.5, vol_scale))
        
        new_vol = current_vol * vol_scale
        # Damping the update: new = old + 0.5 * (target - old)
        new_vol = current_vol + 0.5 * (new_vol - current_vol)
        
        new_vol = max(0.02, min(0.60, new_vol))
        
        print(f"Pass {outer+1}: Max Stress = {max_vm:.1f} MPa, Target = {target_sigma:.1f} MPa. Adj Vol: {current_vol*100:.1f}% -> {new_vol*100:.1f}%")
        
        if abs(new_vol - current_vol) < 0.01:
            current_vol = new_vol
            break
            
        current_vol = new_vol
    
    if progress_callback:
        progress_callback(100, 100)

    # ========== FINAL CLEANUP ==========
    passive_np = passive.to_numpy()[:nx, :ny, :nz]
    is_passive = passive_np == 1  # Road/supports: NEVER touch
    
    # === PHASE 1: BFS Connectivity (remove floating parts) ===
    x_np = x.to_numpy()[:nx, :ny, :nz]
    
    # Seed from foundations (bottom anchor voxels)
    reach = np.zeros((nx, ny, nz), dtype=bool)
    q_bfs = deque()
    for i in range(nx):
        for j in range(ny):
            for k in range(min(3, nz)):
                if ((i >= anchor_pos and i < anchor_pos + anchor_width) or 
                    (i >= nx - anchor_pos - anchor_width and i < nx - anchor_pos)):
                    if x_np[i, j, k] > 0.4 or is_passive[i, j, k]:
                        reach[i, j, k] = True; q_bfs.append((i, j, k))
    # Also seed from ALL passive voxels (road is always "connected")
    for i, j, k in zip(*np.where(is_passive)):
        if not reach[i, j, k]:
            reach[i, j, k] = True; q_bfs.append((i, j, k))
    
    while q_bfs:
        ci, cj, ck = q_bfs.popleft()
        for di, dj, dk in [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]:
            ni, nj, nk = ci+di, cj+dj, ck+dk
            if 0 <= ni < nx and 0 <= nj < ny and 0 <= nk < nz:
                if not reach[ni, nj, nk] and x_np[ni, nj, nk] > 0.4:
                    reach[ni, nj, nk] = True; q_bfs.append((ni, nj, nk))
    
    # Remove everything not reachable (floating) — except passive
    float_removed = np.sum(~reach & ~is_passive & (x_np > 0.4))
    x_np[~reach & ~is_passive] = 0.0
    print(f"Phase 1 (BFS): removed {float_removed} floating voxels")
    
    x_full = np.zeros((MX, MY, MZ), dtype=np.float32)
    x_full[:nx, :ny, :nz] = x_np
    x.from_numpy(x_full)
    enforce_passive_kernel(nx, ny, nz)
    
    # === PHASE 2: Iterative Energy Pruning (remove non-load-bearing parts) ===
    # Uses the EXISTING Taichi FEA solver + calc_sensitivity_kernel.
    # dc[i,j,k] stores -dC/dx which is proportional to strain energy.
    # Voxels with |dc| ≈ 0 contribute nothing to structural integrity.
    
    prune_penal = 3.0
    for prune_iter in range(5):
        x_np = x.to_numpy()[:nx, :ny, :nz]
        n_active_before = int(np.sum((x_np > 0.4) & ~is_passive))
        if n_active_before == 0:
            break
        
        # Run FEA without gravity (external load only)
        filter_vars_kernel(nx, ny, nz, rmin)
        
        # Reset displacements
        u_zero = np.zeros((MX+1, MY+1, MZ+1, 3), dtype=np.float32)
        u.from_numpy(u_zero)
        
        # Copy external forces only (no gravity)
        f_ext_np = f_ext.to_numpy()
        f.from_numpy(f_ext_np)
        
        # Solve CG
        apply_ku_kernel(nx, ny, nz, u, prune_penal)
        rho_cg = cg_init_kernel(nx, ny, nz)
        for _ in range(300):
            if rho_cg < 1e-12: break
            apply_ku_kernel(nx, ny, nz, p, prune_penal)
            pAp_v = cg_step_pAp_kernel(nx, ny, nz)
            alpha_v = rho_cg / (pAp_v + 1e-18)
            rho_new_v = cg_step_ur_kernel(nx, ny, nz, alpha_v)
            cg_step_p_kernel(nx, ny, nz, rho_new_v / (rho_cg + 1e-16))
            rho_cg = rho_new_v
        
        # Compute per-voxel strain energy via Taichi kernel → stored in dc[]
        calc_sensitivity_kernel(nx, ny, nz, prune_penal)
        dc_np = np.abs(dc.to_numpy()[:nx, :ny, :nz])
        
        # Threshold: remove voxels with energy < 1% of median
        active_dc = dc_np[(x_np > 0.4) & ~is_passive]
        if len(active_dc) == 0 or not np.any(active_dc > 0):
            break
        median_dc = np.median(active_dc[active_dc > 0])
        cut = median_dc * 0.01
        
        # Remove low-energy voxels (never passive!)
        remove_mask = (dc_np < cut) & (x_np > 0.4) & ~is_passive
        x_np[remove_mask] = 0.0
        
        # BFS again to remove anything that became disconnected
        reach2 = np.zeros((nx, ny, nz), dtype=bool)
        q2 = deque()
        for i in range(nx):
            for j in range(ny):
                for k in range(min(3, nz)):
                    if ((i >= anchor_pos and i < anchor_pos + anchor_width) or 
                        (i >= nx - anchor_pos - anchor_width and i < nx - anchor_pos)):
                        if x_np[i, j, k] > 0.4 or is_passive[i, j, k]:
                            reach2[i, j, k] = True; q2.append((i, j, k))
        for i, j, k in zip(*np.where(is_passive)):
            if not reach2[i, j, k]:
                reach2[i, j, k] = True; q2.append((i, j, k))
        while q2:
            ci, cj, ck = q2.popleft()
            for di, dj, dk in [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]:
                ni, nj, nk = ci+di, cj+dj, ck+dk
                if 0 <= ni < nx and 0 <= nj < ny and 0 <= nk < nz:
                    if not reach2[ni, nj, nk] and x_np[ni, nj, nk] > 0.4:
                        reach2[ni, nj, nk] = True; q2.append((ni, nj, nk))
        x_np[~reach2 & ~is_passive & (x_np > 0.4)] = 0.0
        
        n_removed = n_active_before - int(np.sum((x_np > 0.4) & ~is_passive))
        print(f"Phase 2 pass {prune_iter+1}: removed {n_removed} non-load-bearing voxels")
        
        if n_removed == 0:
            break
        
        x_full = np.zeros((MX, MY, MZ), dtype=np.float32)
        x_full[:nx, :ny, :nz] = x_np
        x.from_numpy(x_full)
        enforce_passive_kernel(nx, ny, nz)
    
    # === PHASE 3: Absolute Road Protection ===
    x_np = x.to_numpy()[:nx, :ny, :nz]
    x_final = np.zeros((MX, MY, MZ), dtype=np.float32)
    x_final[:nx, :ny, :nz] = x_np
    for i, j, k in zip(*np.where(is_passive)):
        x_final[i, j, k] = 1.0
    x.from_numpy(x_final)
    
    filter_vars_kernel(nx, ny, nz, rmin)
    return xPhys.to_numpy()[:nx, :ny, :nz]


# FORCE MATERIALIZATION ON MAIN THREAD
warmup()
print("Taichi-Motor bereit.")
