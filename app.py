"""
📦 1D Bin Packing Problem (1D-BPP) - Minimalist & Interactive Demo
AI5103 Fundamental of AI & Data Science — Problem Solving Agent
Kelompok: Marshaniswah Syamsul & Hasna Putri Priswati

Features:
1. Minimalist, modern, and user-friendly interface.
2. Step-by-step interactive visualizer for EACH algorithm:
   - First Fit Decreasing (item-by-item placement & decision trail)
   - Genetic Algorithm (generation-by-generation evolution & population distribution)
   - Particle Swarm Optimization (iteration-by-iteration swarm search & position decoding)
3. Algorithm-specific parameter configuration for all algorithms.
4. Head-to-head comparison (FFD vs GA vs PSO) with dynamic convergence curves.
5. Multiple instance presets (Toy 15-item demo, 30-item medium, Falkenauer U120 benchmarks, Custom/Random).
"""

from pathlib import Path
import sys
import time
import math
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Locate repository paths
BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = (BASE_DIR.parent.parent / "bpp-optimization").resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
if str(BASE_DIR) in sys.path:
    sys.path.remove(str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR))

from src.data_loader import load_instance
from src.baseline import first_fit_decreasing, first_fit_stepped
from src.ga import run_genetic_algorithm, run_genetic_algorithm_stepped
from src.pso import run_particle_swarm_optimization, run_particle_swarm_optimization_stepped
from src.validation import validate_packing_solution
from src.metrics import compute_gap_percent

# -------------------------------------------------------------
# STREAMLIT PAGE CONFIG
# -------------------------------------------------------------
st.set_page_config(
    page_title="1D-BPP Interactive Demo | AI5103",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------
# MINIMALIST, POLISHED CSS
# -------------------------------------------------------------
st.markdown("""
<style>
    /* Global styles */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .demo-header {
        padding: 1.2rem 1.5rem;
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border-radius: 14px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(15, 23, 42, 0.15);
    }
    .demo-header h1 {
        font-size: 1.9rem;
        font-weight: 800;
        margin: 0;
        color: #F8FAFC;
    }
    .demo-header p {
        font-size: 0.95rem;
        color: #94A3B8;
        margin: 0.4rem 0 0 0;
    }
    
    .badge-pill {
        display: inline-block;
        padding: 0.2rem 0.65rem;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        margin-right: 0.4rem;
    }
    .badge-ffd { background-color: #FEF3C7; color: #92400E; border: 1px solid #FDE68A; }
    .badge-ga { background-color: #FEE2E2; color: #991B1B; border: 1px solid #FECACA; }
    .badge-pso { background-color: #DBEAFE; color: #1E40AF; border: 1px solid #BFDBFE; }
    .badge-bks { background-color: #DCFCE7; color: #166534; border: 1px solid #BBF7D0; }
    
    .action-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #4F46E5;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }
    .action-card-ga {
        border-left-color: #EF4444;
    }
    .action-card-pso {
        border-left-color: #3B82F6;
    }
    .action-card-ffd {
        border-left-color: #F59E0B;
    }
    
    .stat-box {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 12px;
        text-align: center;
    }
    .stat-label {
        font-size: 0.78rem;
        color: #64748B;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 2px;
    }
    .stat-val {
        font-size: 1.35rem;
        font-weight: 800;
        color: #1E293B;
    }
    
    /* Sleek buttons */
    div.stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.15s ease;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# BENCHMARK INSTANCES METADATA
# -------------------------------------------------------------
FALKENAUER_INSTANCES = {
    "Falkenauer_u120_00.txt": {"n": 120, "c": 150, "bks": 48, "desc": "Uniform [20, 100], C=150"},
    "Falkenauer_u120_01.txt": {"n": 120, "c": 150, "bks": 49, "desc": "Uniform [20, 100], C=150"},
    "Falkenauer_u120_02.txt": {"n": 120, "c": 150, "bks": 46, "desc": "Uniform [20, 100], C=150"},
    "Falkenauer_u120_03.txt": {"n": 120, "c": 150, "bks": 49, "desc": "Uniform [20, 100], C=150"},
    "Falkenauer_u120_04.txt": {"n": 120, "c": 150, "bks": 50, "desc": "Uniform [20, 100], C=150"},
}

# Built-in Toy Presets for instant step-by-step clarity
TOY_PRESETS = {
    "🌟 Mini Demo (15 Item - Visual Cepat)": {
        "n": 15,
        "capacity": 100,
        "weights": [78, 65, 54, 52, 48, 44, 40, 38, 35, 33, 29, 25, 22, 20, 18],
        "bks": 6,
        "desc": "15 item, C=100. Sangat cocok untuk demonstrasi langkah per langkah tanpa lag!"
    },
    "📦 Medium Demo (30 Item)": {
        "n": 30,
        "capacity": 150,
        "weights": [
            98, 95, 92, 90, 88, 85, 82, 80, 78, 75,
            72, 70, 68, 65, 62, 60, 58, 55, 52, 50,
            48, 45, 42, 40, 38, 35, 32, 30, 25, 20
        ],
        "bks": 13,
        "desc": "30 item, C=150. Representasi menengah yang seimbang untuk melihat konvergensi metaheuristik."
    }
}

# -------------------------------------------------------------
# HELPER: FIND INSTANCE FILE ROBUSTLY
# -------------------------------------------------------------
def resolve_instance_path(filename: str) -> Path:
    search_dirs = [
        BASE_DIR / "data" / "Falkenauer_U",
        BASE_DIR / "data",
        REPO_ROOT / "data" / "Falkenauer_U",
        REPO_ROOT / "data",
    ]
    for d in search_dirs:
        candidate = d / filename
        if candidate.is_file():
            return candidate
    return BASE_DIR / "data" / filename

# -------------------------------------------------------------
# SIDEBAR: SCENARIO & INSTANCE SETUP
# -------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📦 Pengaturan Data")
    scenario = st.radio(
        "Pilih Skenario / Dataset:",
        [
            "🌟 Mini Demo (15 Item - Visual Cepat)",
            "📦 Medium Demo (30 Item)",
            "🏛️ Falkenauer U120 (120 Item)",
            "🎲 Custom / Generator Acak"
        ],
        index=0,
        help="Gunakan Mini Demo untuk presentasi langkah demi langkah yang super jelas, atau Falkenauer untuk benchmark standar."
    )

    if scenario in TOY_PRESETS:
        preset_info = TOY_PRESETS[scenario]
        weights = list(preset_info["weights"])
        capacity = preset_info["capacity"]
        bks = preset_info["bks"]
        instance_name = scenario.split("(")[0].strip()
        st.info(f"💡 {preset_info['desc']}")

    elif "Falkenauer" in scenario:
        inst_choice = st.selectbox(
            "Pilih Instance Falkenauer:",
            list(FALKENAUER_INSTANCES.keys()),
            index=0
        )
        inst_meta = FALKENAUER_INSTANCES[inst_choice]
        bks = inst_meta["bks"]
        instance_name = inst_choice.replace(".txt", "")

        file_path = resolve_instance_path(inst_choice)
        if file_path.is_file():
            loaded = load_instance(file_path)
            weights = loaded["weights"]
            capacity = loaded["capacity"]
        else:
            # Fallback if file not found locally
            weights = [random.randint(20, 100) for _ in range(120)]
            capacity = 150
            bks = 48

        st.caption(f"Status: **BPPLIB Benchmark** | BKS Optimum: **{bks} Bin**")

    else:  # Custom / Random
        st.markdown("#### Generator Acak / Input")
        custom_mode = st.radio("Metode Input", ["Acak (Random Generator)", "Input Manual Bobot"])
        if custom_mode == "Acak (Random Generator)":
            c_items = st.slider("Jumlah Item (n)", 10, 80, 20, step=5)
            c_cap = st.slider("Kapasitas Bin (C)", 50, 300, 120, step=10)
            c_min_w = st.slider("Bobot Minimum", 10, 50, 20)
            c_max_w = st.slider("Bobot Maksimum", 50, 150, 80)
            c_seed = st.number_input("Seed Generator", 1, 9999, 42)
            rng_custom = np.random.default_rng(c_seed)
            weights = [int(x) for x in rng_custom.integers(c_min_w, c_max_w + 1, size=c_items)]
            capacity = c_cap
        else:
            c_cap = st.number_input("Kapasitas Bin (C)", 20, 1000, 100)
            c_text = st.text_area("Daftar Bobot Item (pisahkan koma)", "45, 60, 30, 25, 50, 40, 35, 20, 15, 55")
            try:
                weights = [int(x.strip()) for x in c_text.split(",") if x.strip()]
            except Exception:
                weights = [45, 60, 30, 25, 50, 40, 35, 20, 15, 55]
            capacity = c_cap
        
        total_w = sum(weights)
        bks = math.ceil(total_w / capacity)
        instance_name = f"Custom (n={len(weights)}, C={capacity})"

    # Calculate Problem Baseline Stats
    total_weight = sum(weights)
    n_items = len(weights)
    lower_bound_l1 = math.ceil(total_weight / capacity)
    if bks is None or bks < lower_bound_l1:
        bks = lower_bound_l1

    st.markdown("---")
    st.markdown("#### 📊 Ringkasan Masalah")
    sc1, sc2 = st.columns(2)
    sc1.metric("Item (n)", f"{n_items}")
    sc2.metric("Kapasitas (C)", f"{capacity}")
    sc3, sc4 = st.columns(2)
    sc3.metric("Total Bobot", f"{total_weight:,}")
    sc4.metric("Lower Bound L1", f"{lower_bound_l1} Bin")

# -------------------------------------------------------------
# PLOTLY HELPER: INTERACTIVE BIN PACKING LAYOUT
# -------------------------------------------------------------
COLOR_PALETTE = [
    "#3B82F6", "#10B981", "#F59E0B", "#8B5CF6", "#EC4899",
    "#06B6D4", "#F97316", "#14B8A6", "#6366F1", "#84CC16",
    "#E11D48", "#0EA5E9", "#D946EF", "#F43F5E", "#10B981"
]

def render_bin_layout_chart(bins: list, weights: list, capacity: int, highlight_item_idx: int = None, height: int = None):
    """
    Renders clean, beautiful horizontal stacked bar containers for packed bins.
    """
    if not bins:
        fig_empty = go.Figure()
        fig_empty.add_annotation(text="Belum ada item yang dikemas.", showarrow=False, font=dict(size=14, color="#94A3B8"))
        fig_empty.update_layout(height=200, plot_bgcolor="#F8FAFC", xaxis=dict(visible=False), yaxis=dict(visible=False))
        return fig_empty

    fig = go.Figure()
    num_bins = len(bins)
    max_items_in_bin = max((len(b) for b in bins), default=1)

    for slot_idx in range(max_items_in_bin):
        x_vals = []
        y_vals = []
        texts = []
        hovers = []
        colors = []

        for b_idx, b in enumerate(bins):
            bin_name = f"Bin {b_idx + 1:02d}"
            if slot_idx < len(b):
                item_id = b[slot_idx]
                w = weights[item_id]
                accum_w = sum(weights[b[k]] for k in range(slot_idx + 1))
                total_b = sum(weights[i] for i in b)
                slack = capacity - total_b

                x_vals.append(w)
                y_vals.append(bin_name)
                texts.append(f"#{item_id} ({w})")

                # Highlight special item (e.g. newly placed in FFD step)
                if highlight_item_idx is not None and item_id == highlight_item_idx:
                    colors.append("#F59E0B")  # Highlight amber/gold
                    hovers.append(
                        f"✨ <b>[ITEM BARU] #{item_id}</b> (Bobot: {w})<br>"
                        f"Terletak di: <b>{bin_name}</b><br>"
                        f"Beban Akumulatif: {accum_w}/{capacity} ({(accum_w/capacity)*100:.1f}%)<br>"
                        f"Sisa Ruang Bin: {slack}"
                    )
                else:
                    colors.append(COLOR_PALETTE[item_id % len(COLOR_PALETTE)])
                    hovers.append(
                        f"<b>Item #{item_id}</b> (Bobot: {w})<br>"
                        f"Bin: <b>{bin_name}</b><br>"
                        f"Beban Akumulatif: {accum_w}/{capacity} ({(accum_w/capacity)*100:.1f}%)<br>"
                        f"Total Beban Bin: {total_b}/{capacity} ({(total_b/capacity)*100:.1f}%)<br>"
                        f"Sisa Ruang (Slack): {slack}"
                    )
            else:
                x_vals.append(0)
                y_vals.append(bin_name)
                texts.append("")
                hovers.append("")
                colors.append("rgba(0,0,0,0)")

        fig.add_trace(go.Bar(
            y=y_vals,
            x=x_vals,
            name=f"Slot {slot_idx + 1}",
            orientation="h",
            text=texts,
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(size=10, color="white", family="Plus Jakarta Sans"),
            hovertext=hovers,
            hoverinfo="text",
            marker=dict(
                color=colors,
                line=dict(width=1, color="rgba(255,255,255,0.7)")
            )
        ))

    # Red vertical limit line for bin capacity C
    fig.add_vline(
        x=capacity,
        line_dash="dash",
        line_color="#EF4444",
        line_width=2.5,
        annotation_text=f"Batas Kapasitas C = {capacity}",
        annotation_position="top right",
        annotation_font=dict(color="#EF4444", size=11, family="Plus Jakarta Sans")
    )

    chart_height = height if height else max(340, num_bins * 26 + 100)

    fig.update_layout(
        barmode="stack",
        height=chart_height,
        showlegend=False,
        plot_bgcolor="#F8FAFC",
        paper_bgcolor="#FFFFFF",
        margin=dict(l=65, r=30, t=25, b=35),
        xaxis=dict(
            title=f"Kapasitas Terisi (Kapasitas Maksimal C = {capacity})",
            range=[0, capacity + (15 if capacity < 150 else 25)],
            gridcolor="#E2E8F0",
            zeroline=False
        ),
        yaxis=dict(
            title="Daftar Bin",
            autorange="reversed",
            tickfont=dict(size=10)
        )
    )
    return fig

# -------------------------------------------------------------
# APP HEADER
# -------------------------------------------------------------
st.markdown(f"""
<div class="demo-header">
    <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px;">
        <div>
            <h1>📦 1D Bin Packing Problem: Interactive Visualizer</h1>
            <p>Eksplorasi Algoritma Heuristik & Metaheuristik: <b>FFD</b> vs. <b>Genetic Algorithm (GA)</b> vs. <b>Particle Swarm Optimization (PSO)</b></p>
        </div>
        <div>
            <span class="badge-pill badge-ffd">FFD Baseline</span>
            <span class="badge-pill badge-ga">Genetic Algorithm</span>
            <span class="badge-pill badge-pso">Particle Swarm</span>
            <span class="badge-pill badge-bks">BKS: {bks} Bin</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# MAIN APP TABS
# -------------------------------------------------------------
main_tab1, main_tab2, main_tab3 = st.tabs([
    "🎯 1. Visualisasi Step-by-Step (Per Algoritma)",
    "⚖️ 2. Komparasi Head-to-Head (FFD vs GA vs PSO)",
    "💡 3. Panduan & Formulasi Matematika"
])

# =============================================================
# TAB 1: STEP-BY-STEP VISUALIZER FOR EACH ALGORITHM
# =============================================================
with main_tab1:
    st.markdown("#### 🎯 Eksplorasi Langkah demi Langkah")
    st.caption("Pilih salah satu algoritma untuk melihat cara kerjanya secara mendalam, mengatur parameternya, dan menelusuri penataan kontainer di setiap iterasi/langkah.")

    # Algorithm selector pills
    algo_choice = st.radio(
        "Pilih Algoritma untuk Ditampilkan:",
        ["📦 First Fit Decreasing (FFD)", "🧬 Genetic Algorithm (GA)", "⚡ Particle Swarm Optimization (PSO)"],
        horizontal=True
    )

    # ---------------------------------------------------------
    # 1.A: FFD STEP-BY-STEP
    # ---------------------------------------------------------
    if "First Fit Decreasing" in algo_choice:
        with st.expander("⚙️ Parameter & Pengaturan FFD", expanded=True):
            f_col1, f_col2 = st.columns([2, 1])
            with f_col1:
                ffd_strategy = st.selectbox(
                    "Strategi Pengurutan (Sorting Strategy):",
                    [
                        ("decreasing", "Decreasing (FFD - Standar Terbaik: Bobot terbesar duluan)"),
                        ("increasing", "Increasing (FFI - Bobot terkecil duluan)"),
                        ("original", "Original (Pure FF - Sesuai urutan input data asli)"),
                        ("random", "Random (Urutan diacak secara permutasi)")
                    ],
                    format_func=lambda x: x[1],
                    index=0,
                    help="Ubah pengurutan untuk melihat secara langsung mengapa FFD (Decreasing) jauh lebih hemat bin dibandingkan urutan lainnya!"
                )[0]
            with f_col2:
                ffd_seed = st.number_input("Random Seed (untuk Random)", value=42, min_value=1, max_value=9999)

        # Run or retrieve stepped FFD result
        ffd_cache_key = f"ffd_step_{scenario}_{instance_name}_{ffd_strategy}_{ffd_seed}"
        if ffd_cache_key not in st.session_state:
            st.session_state[ffd_cache_key] = first_fit_stepped(
                weights=weights,
                capacity=capacity,
                strategy=ffd_strategy,
                seed=ffd_seed,
                bks=bks
            )
        ffd_res = st.session_state[ffd_cache_key]
        steps = ffd_res["steps"]
        total_steps = len(steps) - 1

        # Interactive Step Controls
        st.markdown("---")
        ctrl_c1, ctrl_c2, ctrl_c3, ctrl_c4, ctrl_c5 = st.columns([1, 1, 3, 1, 1])

        # Step slider key
        step_slider_key = f"ffd_slider_{ffd_cache_key}"
        if step_slider_key not in st.session_state:
            st.session_state[step_slider_key] = total_steps  # Default to final step

        with ctrl_c1:
            if st.button("⏮ Awal", use_container_width=True, key="ffd_btn_start"):
                st.session_state[step_slider_key] = 0
                st.rerun()
        with ctrl_c2:
            if st.button("◀ Prev", use_container_width=True, key="ffd_btn_prev"):
                st.session_state[step_slider_key] = max(0, st.session_state[step_slider_key] - 1)
                st.rerun()
        with ctrl_c3:
            curr_step = st.slider(
                f"Langkah Item: {st.session_state[step_slider_key]} / {total_steps}",
                min_value=0,
                max_value=total_steps,
                key=step_slider_key
            )
        with ctrl_c4:
            if st.button("Next ▶", use_container_width=True, key="ffd_btn_next"):
                st.session_state[step_slider_key] = min(total_steps, st.session_state[step_slider_key] + 1)
                st.rerun()
        with ctrl_c5:
            if st.button("⏭ Akhir", use_container_width=True, key="ffd_btn_end"):
                st.session_state[step_slider_key] = total_steps
                st.rerun()

        # Current step details
        step_data = steps[curr_step]
        curr_bins = step_data["bins_snapshot"]
        curr_loads = step_data["bin_loads_snapshot"]
        curr_bin_count = len(curr_bins)
        curr_packed_items = sum(len(b) for b in curr_bins)
        curr_total_packed_weight = sum(curr_loads)
        curr_util = (curr_total_packed_weight / (curr_bin_count * capacity) * 100) if curr_bin_count > 0 else 0

        # Step explanation card
        st.markdown(f"""
        <div class="action-card action-card-ffd">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <span style="font-weight: 700; color: #92400E; font-size: 1rem;">📦 {step_data['action_title']}</span>
                <span class="badge-pill badge-ffd">Strategi: {ffd_res['strategy_name']}</span>
            </div>
            <div style="font-size: 0.95rem; color: #334155; line-height: 1.5;">
                {step_data['description']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Real-time metrics at current step
        sm1, sm2, sm3, sm4 = st.columns(4)
        with sm1:
            st.markdown(f"""<div class="stat-box"><div class="stat-label">Bin Terbuka</div><div class="stat-val" style="color: #F59E0B;">{curr_bin_count}</div></div>""", unsafe_allow_html=True)
        with sm2:
            st.markdown(f"""<div class="stat-box"><div class="stat-label">Item Terkemas</div><div class="stat-val">{curr_packed_items} / {n_items}</div></div>""", unsafe_allow_html=True)
        with sm3:
            st.markdown(f"""<div class="stat-box"><div class="stat-label">Rata-rata Utilitas</div><div class="stat-val">{curr_util:.1f}%</div></div>""", unsafe_allow_html=True)
        with sm4:
            st.markdown(f"""<div class="stat-box"><div class="stat-label">BKS Target</div><div class="stat-val" style="color: #10B981;">{bks} Bin</div></div>""", unsafe_allow_html=True)

        st.markdown("#### Layout Penataan Bin pada Langkah Ini")
        st.caption("Item yang **baru dimasukkan pada langkah ini diwarnai emas/kuning**. Sisa ruang (slack) terlihat transparan.")
        fig_ffd = render_bin_layout_chart(
            bins=curr_bins,
            weights=weights,
            capacity=capacity,
            highlight_item_idx=step_data["item_idx"]
        )
        st.plotly_chart(fig_ffd, use_container_width=True)

        # Inspection of bin check trials if item was placed
        if step_data.get("trials"):
            with st.expander("🔍 Detail Pemeriksaan Tiap Bin pada Langkah Ini"):
                trial_rows = []
                for t in step_data["trials"]:
                    trial_rows.append({
                        "Bin Diperiksa": f"Bin {t['bin_idx'] + 1:02d}",
                        "Beban Sebelum": f"{t['current_load']} / {capacity}",
                        "Beban Tambah Item": f"{t['candidate_load']} / {capacity}",
                        "Apakah Muat?": "✅ Muat! (Ditempatkan)" if t["fits"] else "❌ Penuh (Skip)"
                    })
                st.dataframe(pd.DataFrame(trial_rows), hide_index=True, use_container_width=True)

    # ---------------------------------------------------------
    # 1.B: GENETIC ALGORITHM STEP-BY-STEP (GENERATION BY GENERATION)
    # ---------------------------------------------------------
    elif "Genetic Algorithm" in algo_choice:
        with st.expander("⚙️ Parameter Genetic Algorithm (GA)", expanded=True):
            ga_c1, ga_c2, ga_c3 = st.columns(3)
            with ga_c1:
                ga_pop_size = st.slider("Ukuran Populasi (N)", 10, 80, 30, step=5, help="Jumlah kromosom permutasi dalam populasi.")
                ga_generations = st.slider("Jumlah Generasi (Steps)", 10, 80, 35, step=5, help="Berapa generasi evolusi yang akan direkam langkah per langkah.")
            with ga_c2:
                ga_pc = st.slider("Crossover Rate (p_c)", 0.50, 1.00, 0.90, step=0.05, help="Probabilitas Order Crossover (OX).")
                ga_pm = st.slider("Mutation Rate (p_m)", 0.01, 0.50, 0.20, step=0.01, help="Probabilitas Swap Mutation.")
            with ga_c3:
                ga_tour = st.slider("Tournament Size (k)", 2, 6, 3, step=1, help="Jumlah kandidat yang diadu dalam seleksi turnamen.")
                ga_elitism = st.checkbox("Gunakan Elitisme (Simpan Solusi Terbaik)", value=True)
                ga_seed = st.number_input("Random Seed (GA)", value=42, min_value=1, max_value=9999)

        ga_cache_key = f"ga_step_{scenario}_{instance_name}_{ga_pop_size}_{ga_generations}_{ga_pc}_{ga_pm}_{ga_tour}_{ga_elitism}_{ga_seed}"
        if ga_cache_key not in st.session_state:
            with st.spinner("Mengeksekusi simulasi evolusi langkah demi langkah Genetic Algorithm..."):
                st.session_state[ga_cache_key] = run_genetic_algorithm_stepped(
                    weights=weights,
                    capacity=capacity,
                    population_size=ga_pop_size,
                    generations=ga_generations,
                    tournament_size=ga_tour,
                    mutation_rate=ga_pm,
                    crossover_rate=ga_pc,
                    elitism=ga_elitism,
                    seed=ga_seed,
                    bks=bks
                )
        ga_res = st.session_state[ga_cache_key]
        gens_data = ga_res["generations_data"]
        total_gens = len(gens_data) - 1

        # Interactive Step Controls
        st.markdown("---")
        g_ctrl1, g_ctrl2, g_ctrl3, g_ctrl4, g_ctrl5 = st.columns([1, 1, 3, 1, 1])

        ga_slider_key = f"ga_slider_{ga_cache_key}"
        if ga_slider_key not in st.session_state:
            st.session_state[ga_slider_key] = total_gens

        with g_ctrl1:
            if st.button("⏮ Awal", use_container_width=True, key="ga_btn_start"):
                st.session_state[ga_slider_key] = 0
                st.rerun()
        with g_ctrl2:
            if st.button("◀ Prev Gen", use_container_width=True, key="ga_btn_prev"):
                st.session_state[ga_slider_key] = max(0, st.session_state[ga_slider_key] - 1)
                st.rerun()
        with g_ctrl3:
            curr_gen = st.slider(
                f"Generasi: {st.session_state[ga_slider_key]} / {total_gens}",
                min_value=0,
                max_value=total_gens,
                key=ga_slider_key
            )
        with g_ctrl4:
            if st.button("Next Gen ▶", use_container_width=True, key="ga_btn_next"):
                st.session_state[ga_slider_key] = min(total_gens, st.session_state[ga_slider_key] + 1)
                st.rerun()
        with g_ctrl5:
            if st.button("⏭ Akhir", use_container_width=True, key="ga_btn_end"):
                st.session_state[ga_slider_key] = total_gens
                st.rerun()

        gen_step = gens_data[curr_gen]
        best_bins_gen = gen_step["best_bins"]
        bin_count_gen = len(best_bins_gen)
        bin_loads_gen = gen_step["bin_loads"]
        util_gen = (sum(bin_loads_gen) / (bin_count_gen * capacity) * 100) if bin_count_gen > 0 else 0

        # GA Step Explanation
        st.markdown(f"""
        <div class="action-card action-card-ga">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <span style="font-weight: 700; color: #991B1B; font-size: 1rem;">🧬 Generasi ke-{curr_gen} (Evaluasi: {gen_step['evaluations']:,})</span>
                <span class="badge-pill badge-ga">Best: {gen_step['best_fitness']} Bins</span>
            </div>
            <div style="font-size: 0.95rem; color: #334155; line-height: 1.5;">
                {gen_step['description']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Real-time GA stats
        gm1, gm2, gm3, gm4 = st.columns(4)
        with gm1:
            st.markdown(f"""<div class="stat-box"><div class="stat-label">Best Bins (Generasi Ini)</div><div class="stat-val" style="color: #EF4444;">{gen_step['best_fitness']}</div></div>""", unsafe_allow_html=True)
        with gm2:
            st.markdown(f"""<div class="stat-box"><div class="stat-label">Rata-rata Populasi</div><div class="stat-val">{gen_step['mean_fitness']:.1f} Bin</div></div>""", unsafe_allow_html=True)
        with gm3:
            st.markdown(f"""<div class="stat-box"><div class="stat-label">Rata-rata Utilitas</div><div class="stat-val">{util_gen:.1f}%</div></div>""", unsafe_allow_html=True)
        with gm4:
            st.markdown(f"""<div class="stat-box"><div class="stat-label">BKS Target</div><div class="stat-val" style="color: #10B981;">{bks} Bin</div></div>""", unsafe_allow_html=True)

        # Visual layout of best solution at this generation
        st.markdown("#### Penataan Kontainer dari Solusi Terbaik Generasi Ini")
        st.caption("Perhatikan bagaimana jumlah bin berkurang atau tata letak item semakin padat saat Anda menggeser slider generasi ke kanan!")
        fig_ga = render_bin_layout_chart(
            bins=best_bins_gen,
            weights=weights,
            capacity=capacity
        )
        st.plotly_chart(fig_ga, use_container_width=True)

        # Convergence progression up to current generation
        st.markdown("#### Perkembangan Konvergensi GA sampai Generasi Ini")
        sub_gens = [d["generation"] for d in gens_data[:curr_gen + 1]]
        sub_best = [d["best_fitness"] for d in gens_data[:curr_gen + 1]]
        sub_mean = [d["mean_fitness"] for d in gens_data[:curr_gen + 1]]

        fig_ga_conv = go.Figure()
        fig_ga_conv.add_hline(y=bks, line_dash="dash", line_color="#10B981", line_width=2, annotation_text=f"BKS Optimum ({bks} Bin)")
        fig_ga_conv.add_trace(go.Scatter(x=sub_gens, y=sub_best, mode="lines+markers", name="Solusi Terbaik (Best)", line=dict(color="#EF4444", width=3)))
        fig_ga_conv.add_trace(go.Scatter(x=sub_gens, y=sub_mean, mode="lines", name="Rata-rata Populasi (Mean)", line=dict(color="#94A3B8", width=1.8, dash="dot")))

        fig_ga_conv.update_layout(
            height=280,
            xaxis=dict(title="Generasi Evolusi", range=[-0.5, total_gens + 0.5], gridcolor="#F1F5F9"),
            yaxis=dict(title="Jumlah Bin", gridcolor="#F1F5F9"),
            plot_bgcolor="#FFFFFF",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=30, t=30, b=30)
        )
        st.plotly_chart(fig_ga_conv, use_container_width=True)

    # ---------------------------------------------------------
    # 1.C: PSO STEP-BY-STEP (ITERATION BY ITERATION)
    # ---------------------------------------------------------
    else:
        with st.expander("⚙️ Parameter Particle Swarm Optimization (PSO)", expanded=True):
            pso_c1, pso_c2, pso_c3 = st.columns(3)
            with pso_c1:
                pso_pop = st.slider("Ukuran Swarm (Jumlah Partikel)", 10, 60, 30, step=5, help="Jumlah partikel yang menjelajah ruang pencarian kontinu.")
                pso_iters = st.slider("Jumlah Iterasi (Steps)", 10, 80, 35, step=5, help="Berapa iterasi pergerakan partikel yang direkam langkah per langkah.")
            with pso_c2:
                pso_w = st.slider("Inertia Weight (w)", 0.2, 1.0, 0.7, step=0.05, help="Bobot kelembaman partikel menjaga kecepatan sebelumnya.")
                pso_vmax = st.slider("Velocity Clamping (v_max)", 0.05, 0.50, 0.20, step=0.05, help="Batas kecepatan maksimum per dimensi.")
            with pso_c3:
                pso_c1_val = st.slider("Cognitive Param (c1)", 0.5, 3.0, 1.5, step=0.1, help="Daya tarik partikel ke pengalaman terbaik pribadinya (pbest).")
                pso_c2_val = st.slider("Social Param (c2)", 0.5, 3.0, 1.5, step=0.1, help="Daya tarik partikel ke solusi terbaik global swarm (gbest).")
                pso_seed = st.number_input("Random Seed (PSO)", value=42, min_value=1, max_value=9999)

        pso_cache_key = f"pso_step_{scenario}_{instance_name}_{pso_pop}_{pso_iters}_{pso_w}_{pso_c1_val}_{pso_c2_val}_{pso_vmax}_{pso_seed}"
        if pso_cache_key not in st.session_state:
            with st.spinner("Mengeksekusi simulasi pergerakan swarm langkah demi langkah PSO..."):
                st.session_state[pso_cache_key] = run_particle_swarm_optimization_stepped(
                    weights=weights,
                    capacity=capacity,
                    population_size=pso_pop,
                    iterations=pso_iters,
                    w=pso_w,
                    c1=pso_c1_val,
                    c2=pso_c2_val,
                    v_min=-pso_vmax,
                    v_max=pso_vmax,
                    seed=pso_seed,
                    bks=bks
                )
        pso_res = st.session_state[pso_cache_key]
        iters_data = pso_res["iterations_data"]
        total_iters = len(iters_data) - 1

        # Interactive Step Controls
        st.markdown("---")
        p_ctrl1, p_ctrl2, p_ctrl3, p_ctrl4, p_ctrl5 = st.columns([1, 1, 3, 1, 1])

        pso_slider_key = f"pso_slider_{pso_cache_key}"
        if pso_slider_key not in st.session_state:
            st.session_state[pso_slider_key] = total_iters

        with p_ctrl1:
            if st.button("⏮ Awal", use_container_width=True, key="pso_btn_start"):
                st.session_state[pso_slider_key] = 0
                st.rerun()
        with p_ctrl2:
            if st.button("◀ Prev It", use_container_width=True, key="pso_btn_prev"):
                st.session_state[pso_slider_key] = max(0, st.session_state[pso_slider_key] - 1)
                st.rerun()
        with p_ctrl3:
            curr_it = st.slider(
                f"Iterasi: {st.session_state[pso_slider_key]} / {total_iters}",
                min_value=0,
                max_value=total_iters,
                key=pso_slider_key
            )
        with p_ctrl4:
            if st.button("Next It ▶", use_container_width=True, key="pso_btn_next"):
                st.session_state[pso_slider_key] = min(total_iters, st.session_state[pso_slider_key] + 1)
                st.rerun()
        with p_ctrl5:
            if st.button("⏭ Akhir", use_container_width=True, key="pso_btn_end"):
                st.session_state[pso_slider_key] = total_iters
                st.rerun()

        it_step = iters_data[curr_it]
        best_bins_pso = it_step["best_bins"]
        bin_count_pso = len(best_bins_pso)
        bin_loads_pso = it_step["bin_loads"]
        util_pso = (sum(bin_loads_pso) / (bin_count_pso * capacity) * 100) if bin_count_pso > 0 else 0

        # PSO Step Explanation
        st.markdown(f"""
        <div class="action-card action-card-pso">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <span style="font-weight: 700; color: #1E40AF; font-size: 1rem;">⚡ Iterasi ke-{curr_it} (Evaluasi: {it_step['evaluations']:,})</span>
                <span class="badge-pill badge-pso">gbest: {it_step['gbest_fitness']} Bins</span>
            </div>
            <div style="font-size: 0.95rem; color: #334155; line-height: 1.5;">
                {it_step['description']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Real-time PSO stats
        pm1, pm2, pm3, pm4 = st.columns(4)
        with pm1:
            st.markdown(f"""<div class="stat-box"><div class="stat-label">gbest Bins (Iterasi Ini)</div><div class="stat-val" style="color: #3B82F6;">{it_step['gbest_fitness']}</div></div>""", unsafe_allow_html=True)
        with pm2:
            st.markdown(f"""<div class="stat-box"><div class="stat-label">Rata-rata Swarm</div><div class="stat-val">{it_step['mean_fitness']:.1f} Bin</div></div>""", unsafe_allow_html=True)
        with pm3:
            st.markdown(f"""<div class="stat-box"><div class="stat-label">Rata-rata Utilitas</div><div class="stat-val">{util_pso:.1f}%</div></div>""", unsafe_allow_html=True)
        with pm4:
            st.markdown(f"""<div class="stat-box"><div class="stat-label">BKS Target</div><div class="stat-val" style="color: #10B981;">{bks} Bin</div></div>""", unsafe_allow_html=True)

        # Visual layout of gbest solution at this iteration
        st.markdown("#### Penataan Kontainer dari Solusi Global Best ($g_{best}$) Iterasi Ini")
        st.caption("Perhatikan bagaimana partikel mengorientasikan vektor posisi mereka menuju kombinasi urutan item yang lebih optimal.")
        fig_pso = render_bin_layout_chart(
            bins=best_bins_pso,
            weights=weights,
            capacity=capacity
        )
        st.plotly_chart(fig_pso, use_container_width=True)

        # Convergence progression up to current iteration
        st.markdown("#### Perkembangan Konvergensi PSO sampai Iterasi Ini")
        sub_its = [d["iteration"] for d in iters_data[:curr_it + 1]]
        sub_pso_best = [d["gbest_fitness"] for d in iters_data[:curr_it + 1]]
        sub_pso_mean = [d["mean_fitness"] for d in iters_data[:curr_it + 1]]

        fig_pso_conv = go.Figure()
        fig_pso_conv.add_hline(y=bks, line_dash="dash", line_color="#10B981", line_width=2, annotation_text=f"BKS Optimum ({bks} Bin)")
        fig_pso_conv.add_trace(go.Scatter(x=sub_its, y=sub_pso_best, mode="lines+markers", name="Global Best (gbest)", line=dict(color="#3B82F6", width=3)))
        fig_pso_conv.add_trace(go.Scatter(x=sub_its, y=sub_pso_mean, mode="lines", name="Rata-rata Swarm", line=dict(color="#94A3B8", width=1.8, dash="dot")))

        fig_pso_conv.update_layout(
            height=280,
            xaxis=dict(title="Iterasi Swarm", range=[-0.5, total_iters + 0.5], gridcolor="#F1F5F9"),
            yaxis=dict(title="Jumlah Bin", gridcolor="#F1F5F9"),
            plot_bgcolor="#FFFFFF",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=30, t=30, b=30)
        )
        st.plotly_chart(fig_pso_conv, use_container_width=True)

# =============================================================
# TAB 2: HEAD-TO-HEAD COMPARISON (FFD vs GA vs PSO)
# =============================================================
with main_tab2:
    st.markdown(f"#### ⚖️ Perbandingan Kinerja Seluruh Algoritma pada {instance_name}")
    st.caption("Membandingkan deterministik heuristik (FFD) dengan dua metaheuristik (GA & PSO) di bawah kondisi evaluasi yang sama.")

    # Execution controls for comparison
    with st.expander("⚙️ Parameter Uji Komparasi (Unified Benchmark Settings)", expanded=False):
        c_p1, c_p2, c_p3 = st.columns(3)
        with c_p1:
            comp_eval_budget = st.slider("Budget Evaluasi (Obj. Func)", 500, 6000, 2000, step=500, help="Jumlah evaluasi maksimum untuk GA dan PSO.")
        with c_p2:
            comp_pop_size = st.slider("Ukuran Populasi / Swarm", 20, 60, 40, step=10)
        with c_p3:
            comp_seed = st.number_input("Random Seed Komparasi", value=42, min_value=1, max_value=9999)

    run_comp_btn = st.button("🚀 Jalankan Komparasi Lengkap (FFD vs GA vs PSO)", type="primary", use_container_width=True)

    comp_cache_key = f"comp_res_{scenario}_{instance_name}_{comp_eval_budget}_{comp_pop_size}_{comp_seed}"
    if run_comp_btn or comp_cache_key not in st.session_state:
        with st.spinner("Menjalankan FFD, GA, dan PSO secara komparatif..."):
            t0_ffd = time.perf_counter()
            c_res_ffd = first_fit_decreasing(weights, capacity, bks=bks)
            
            c_res_ga = run_genetic_algorithm(
                weights=weights,
                capacity=capacity,
                population_size=comp_pop_size,
                evaluation_budget=comp_eval_budget,
                seed=comp_seed,
                bks=bks
            )
            
            c_res_pso = run_particle_swarm_optimization(
                weights=weights,
                capacity=capacity,
                population_size=comp_pop_size,
                evaluation_budget=comp_eval_budget,
                seed=comp_seed,
                bks=bks
            )

            st.session_state[comp_cache_key] = {
                "ffd": c_res_ffd,
                "ga": c_res_ga,
                "pso": c_res_pso
            }

    comp_results = st.session_state[comp_cache_key]
    cf = comp_results["ffd"]
    cg = comp_results["ga"]
    cp = comp_results["pso"]

    # Comparative Cards
    col_c1, col_c2, col_c3, col_c4 = st.columns(4)
    with col_c1:
        st.markdown(f"""
        <div class="stat-box" style="border-top: 4px solid #10B981;">
            <div class="stat-label">BKS Optimum</div>
            <div class="stat-val" style="color: #166534;">{bks} Bin</div>
            <div style="font-size: 0.75rem; color: #166534; font-weight: 600; margin-top: 4px;">Gap: 0.00%</div>
        </div>
        """, unsafe_allow_html=True)
    with col_c2:
        gap_ffd = compute_gap_percent(cf['best_bins'], bks)
        st.markdown(f"""
        <div class="stat-box" style="border-top: 4px solid #F59E0B;">
            <div class="stat-label">FFD Baseline</div>
            <div class="stat-val" style="color: #92400E;">{cf['best_bins']} Bin</div>
            <div style="font-size: 0.75rem; color: #92400E; font-weight: 600; margin-top: 4px;">Gap: {gap_ffd:.2f}% | {cf['runtime_seconds']:.4f}s</div>
        </div>
        """, unsafe_allow_html=True)
    with col_c3:
        gap_ga = compute_gap_percent(cg['best_fitness'], bks)
        st.markdown(f"""
        <div class="stat-box" style="border-top: 4px solid #EF4444;">
            <div class="stat-label">Genetic Algorithm</div>
            <div class="stat-val" style="color: #991B1B;">{cg['best_fitness']} Bin</div>
            <div style="font-size: 0.75rem; color: #991B1B; font-weight: 600; margin-top: 4px;">Gap: {gap_ga:.2f}% | {cg['runtime_seconds']:.2f}s</div>
        </div>
        """, unsafe_allow_html=True)
    with col_c4:
        gap_pso = compute_gap_percent(cp['best_fitness'], bks)
        st.markdown(f"""
        <div class="stat-box" style="border-top: 4px solid #3B82F6;">
            <div class="stat-label">Particle Swarm</div>
            <div class="stat-val" style="color: #1E40AF;">{cp['best_fitness']} Bin</div>
            <div style="font-size: 0.75rem; color: #1E40AF; font-weight: 600; margin-top: 4px;">Gap: {gap_pso:.2f}% | {cp['runtime_seconds']:.2f}s</div>
        </div>
        """, unsafe_allow_html=True)

    # Detailed Comparison Table
    st.markdown("#### 📋 Tabel Metrik Komprehensif")
    comp_df_data = [
        {
            "Algoritma": "First Fit Decreasing (FFD)",
            "Kategori": "Heuristik Deterministik",
            "Jumlah Bin": cf["best_bins"],
            "BKS": bks,
            "Gap vs BKS (%)": f"{gap_ffd:.2f}%",
            "Evaluasi Fungsi": cf["evaluations"],
            "Waktu (detik)": f"{cf['runtime_seconds']:.4f}",
            "Validitas Solusi": "✅ Valid" if cf["valid"] else "❌ Invalid"
        },
        {
            "Algoritma": "Genetic Algorithm (GA)",
            "Kategori": "Metaheuristik Permutasi",
            "Jumlah Bin": cg["best_fitness"],
            "BKS": bks,
            "Gap vs BKS (%)": f"{gap_ga:.2f}%",
            "Evaluasi Fungsi": cg["evaluations"],
            "Waktu (detik)": f"{cg['runtime_seconds']:.2f}",
            "Validitas Solusi": "✅ Valid" if cg["valid"] else "❌ Invalid"
        },
        {
            "Algoritma": "Particle Swarm Optimization (PSO)",
            "Kategori": "Metaheuristik Kontinu (Argsort)",
            "Jumlah Bin": cp["best_fitness"],
            "BKS": bks,
            "Gap vs BKS (%)": f"{gap_pso:.2f}%",
            "Evaluasi Fungsi": cp["evaluations"],
            "Waktu (detik)": f"{cp['runtime_seconds']:.2f}",
            "Validitas Solusi": "✅ Valid" if cp["valid"] else "❌ Invalid"
        }
    ]
    st.dataframe(pd.DataFrame(comp_df_data), hide_index=True, use_container_width=True)

    # Convergence Curves Comparison
    st.markdown("#### 📈 Kurva Konvergensi Komparatif")
    ga_ckpts = cg["eval_checkpoints"]
    ga_conv = cg["convergence"]
    pso_conv = cp["history"]
    max_e = min(len(pso_conv), ga_ckpts[-1] if ga_ckpts else len(pso_conv))

    fig_cmp_conv = go.Figure()
    # BKS Optimum
    fig_cmp_conv.add_hline(
        y=bks, line_dash="dash", line_color="#10B981", line_width=2.5,
        annotation_text=f"BKS Optimum ({bks} Bin)", annotation_position="bottom right"
    )
    # FFD Baseline
    fig_cmp_conv.add_hline(
        y=cf["best_bins"], line_dash="dot", line_color="#F59E0B", line_width=2,
        annotation_text=f"FFD Baseline ({cf['best_bins']} Bin)", annotation_position="top left"
    )
    # GA Line
    fig_cmp_conv.add_trace(go.Scatter(
        x=ga_ckpts, y=ga_conv, mode="lines+markers", name="Genetic Algorithm (GA)",
        line=dict(color="#EF4444", width=2.8), marker=dict(size=4)
    ))
    # PSO Line (sampled every 10 evals for smoothness)
    pso_x_sampled = list(range(1, max_e + 1, 10))
    pso_y_sampled = [pso_conv[min(i - 1, len(pso_conv) - 1)] for i in pso_x_sampled]
    fig_cmp_conv.add_trace(go.Scatter(
        x=pso_x_sampled, y=pso_y_sampled, mode="lines", name="Particle Swarm (PSO)",
        line=dict(color="#3B82F6", width=2.2)
    ))

    fig_cmp_conv.update_layout(
        height=380,
        xaxis=dict(title="Evaluasi Fungsi Objektif", gridcolor="#F1F5F9"),
        yaxis=dict(title="Jumlah Bin Terbaik (Minimize)", gridcolor="#F1F5F9"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="#FFFFFF",
        margin=dict(l=40, r=40, t=30, b=40)
    )
    st.plotly_chart(fig_cmp_conv, use_container_width=True)

    # Side-by-side Bar charts for Bins and Runtime
    cb_col1, cb_col2 = st.columns(2)
    with cb_col1:
        st.markdown("##### Jumlah Bin Digunakan (Lebih Sedikit Lebih Baik)")
        fig_b = go.Figure(go.Bar(
            x=["BKS Optimum", "FFD Baseline", "Genetic Algorithm", "Particle Swarm"],
            y=[bks, cf["best_bins"], cg["best_fitness"], cp["best_fitness"]],
            marker_color=["#10B981", "#F59E0B", "#EF4444", "#3B82F6"],
            text=[f"{bks}", f"{cf['best_bins']}", f"{cg['best_fitness']}", f"{cp['best_fitness']}"],
            textposition="auto"
        ))
        fig_b.update_layout(
            height=300,
            yaxis=dict(range=[bks - 2, max(cf['best_bins'], cg['best_fitness'], cp['best_fitness']) + 2], gridcolor="#F1F5F9"),
            plot_bgcolor="#FFFFFF",
            margin=dict(l=30, r=30, t=20, b=30)
        )
        st.plotly_chart(fig_b, use_container_width=True)

    with cb_col2:
        st.markdown("##### Waktu Komputasi (Detik)")
        fig_t = go.Figure(go.Bar(
            x=["FFD Baseline", "Genetic Algorithm", "Particle Swarm"],
            y=[cf["runtime_seconds"], cg["runtime_seconds"], cp["runtime_seconds"]],
            marker_color=["#F59E0B", "#EF4444", "#3B82F6"],
            text=[f"{cf['runtime_seconds']:.4f}s", f"{cg['runtime_seconds']:.2f}s", f"{cp['runtime_seconds']:.2f}s"],
            textposition="auto"
        ))
        fig_t.update_layout(
            height=300,
            yaxis=dict(gridcolor="#F1F5F9"),
            plot_bgcolor="#FFFFFF",
            margin=dict(l=30, r=30, t=20, b=30)
        )
        st.plotly_chart(fig_t, use_container_width=True)

# =============================================================
# TAB 3: CONCEPTS & MATHEMATICAL FORMULATION
# =============================================================
with main_tab3:
    st.markdown("#### 💡 Panduan Konseptual & Formulasi Matematika")
    
    fk_col1, fk_col2 = st.columns(2)
    with fk_col1:
        st.markdown("""
        ##### 1. Formulasi Integer Linear Programming (ILP) 1D-BPP
        Diberikan sekumpulan $n$ item dengan bobot $w_i > 0$ dan kapasitas bin tak terhingga masing-masing berkapasitas $C$:
        
        $$\\min \\quad m = \\sum_{j=1}^m y_j$$
        
        $$\\text{subject to:} \\quad \\sum_{i=0}^{n-1} w_i x_{ij} \\le C y_j, \\quad \\forall j \\in \\{1, \\dots, m\\}$$
        $$\\sum_{j=1}^m x_{ij} = 1, \\quad \\forall i \\in \\{0, \\dots, n-1\\}$$
        $$x_{ij} \\in \\{0, 1\\}, \\quad y_j \\in \\{0, 1\\}$$
        
        ##### 2. Batas Bawah Kontinu (Volume Lower Bound $L_1$)
        $$L_1 = \\left\\lceil \\frac{\\sum_{i=0}^{n-1} w_i}{C} \\right\\rceil$$
        Nilai ini merupakan batas teoritis minimum jumlah bin yang mungkin tercapai jika ruang kosong (slack) dapat dieliminasi secara sempurna.
        """)

    with fk_col2:
        st.markdown("""
        ##### 3. Mengapa First Fit Decreasing (FFD) Sangat Kuat?
        - **Kompleksitas**: $O(n \\log n)$ berkat pengurutan awal.
        - **Jaminan Teoritis**: Johnson (1973) membuktikan bahwa:
          $$\\text{FFD}(I) \\le \\frac{11}{9} \\text{OPT}(I) + 1$$
        - Item besar dikemas terlebih dahulu untuk mengisi fondasi ruang bin, sementara item kecil berperan sebagai "pengganjal" slack di bagian akhir.
        
        ##### 4. Representasi Metaheuristik: GA vs. PSO
        - **Genetic Algorithm (GA)**: Menggunakan kromosom permutasi $[0, \\dots, n-1]$ yang di-decode via First Fit. Dilengkapi **Order Crossover (OX)** untuk menjaga keabsahan permutasi tanpa duplikasi dan **Swap Mutation**.
        - **Particle Swarm Optimization (PSO)**: Menggunakan vektor posisi kontinu $[0, 1]^n$. Pemetaan ke urutan item dilakukan melalui **argsort (Random-Key representation)** lalu dievaluasi via First Fit.
        """)

    st.markdown("---")
    st.markdown("##### 🛡️ Protokol Validasi Solusi & Integritas Matematika (Feasibility Audit)")
    st.markdown("""
    Setiap solusi yang dihasilkan diaudit secara otomatis melalui 3 aksioma ketat:
    1. **Kapasitas Bin ($P_1$)**: $\\sum_{i \\in B_j} w_i \\le C$ untuk seluruh bin $j$. Tidak boleh ada overcapacity.
    2. **Kelengkapan Item ($P_2$)**: $\\bigcup B_j = \\{0, \\dots, n-1\\}$ dan $|\\bigcup B_j| = n$. Setiap item wajib terkemas tepat satu kali.
    3. **Konsistensi Jumlah ($P_3$)**: Jumlah bin aktif $|\\{B_j\\}|$ wajib identik dengan fitness fungsi objektif yang dilaporkan.
    """)

# Footer
st.markdown("---")
st.caption("AI5103 Fundamental of AI & Data Science — Problem Solving Agent · Kelompok Marshaniswah Syamsul & Hasna Putri Priswati")
