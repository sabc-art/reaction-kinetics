"""Reaction Kinetics Simulator + Data Fitter (Streamlit).

Run with:  python -m streamlit run kinetics.py
"""
import base64
import io
import re
from fractions import Fraction
from math import gcd

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
import matplotlib.pyplot as plt


# =====================================================================
# HELPER FUNCTIONS (one small function for each calculation)
# =====================================================================

def sci_value(coef, exp):
    """Scientific notation: coefficient x 10^exponent."""
    return coef * 10.0 ** exp


def sci_input(label, default_coef, default_exp, key):
    """Two boxes (number and power of 10). Returns the final value."""
    st.write(f"**{label}**")
    c1, c2, c3 = st.columns([2, 2, 3])
    coef = c1.number_input("Number", value=float(default_coef),
                           format="%.4f", key=key + "_coef")
    exp = c2.number_input("Power of 10", value=int(default_exp),
                          step=1, key=key + "_exp")
    value = sci_value(coef, exp)
    c3.write("Value")
    c3.write(f"**{value:.4g}**")
    return value


def concentration_curves(k, C0, t):
    """Concentration vs time for 0th, 1st and 2nd order reactions."""
    return {
        0: np.maximum(C0 - k * t, 0),      # C = C0 - k t
        1: C0 * np.exp(-k * t),            # C = C0 e^(-k t)
        2: C0 / (1 + k * C0 * t),          # 1/C = 1/C0 + k t
    }


def half_lives(k, C0):
    """Half-life for each reaction order."""
    return {
        0: C0 / (2 * k),
        1: np.log(2) / k,
        2: 1 / (k * C0),
    }


def figure_bytes(fig, fmt):
    """Save a figure in memory (no file on the computer) and return bytes."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format=fmt, bbox_inches="tight", dpi=200)
    return buffer.getvalue()


def download_buttons(fig, name, key):
    """Two buttons: download the graph as PDF or JPG."""
    col1, col2 = st.columns(2)
    col1.download_button("Download graph as PDF", figure_bytes(fig, "pdf"),
                         file_name=name + ".pdf", mime="application/pdf",
                         key=key + "_pdf")
    col2.download_button("Download graph as JPG", figure_bytes(fig, "jpg"),
                         file_name=name + ".jpg", mime="image/jpeg",
                         key=key + "_jpg")


def linearize(order, C):
    """Turn concentration into the value that gives a straight line."""
    if order == 0:
        return C                 # C vs t
    if order == 1:
        return np.log(C)         # ln(C) vs t
    return 1.0 / C               # 1/C vs t


def fit_order(order, t, C):
    """Straight-line fit for one reaction order. Returns slope, k, R2."""
    y = linearize(order, C)
    slope, intercept = np.polyfit(t, y, 1)
    y_fit = slope * t + intercept
    ss_res = np.sum((y - y_fit) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    k = -slope if order in (0, 1) else slope   # 0th and 1st have negative slope
    return {"order": order, "slope": slope, "intercept": intercept,
            "k": k, "r2": r2}


def model_curve(order, slope, intercept, t):
    """Fitted concentration curve, drawn on the C vs t graph."""
    y = slope * t + intercept
    if order == 0:
        return y
    if order == 1:
        return np.exp(y)
    with np.errstate(divide="ignore", invalid="ignore"):
        c = 1.0 / y
    c[y <= 0] = np.nan
    return c


def k_units(order, time_unit, conc_unit):
    """Units of k depend on the reaction order."""
    amount, volume = conc_unit.split("/")
    if order == 0:
        return f"{amount}/({volume}·{time_unit})"
    if order == 1:
        return f"1/{time_unit}"
    return f"{volume}/({amount}·{time_unit})"


def rate_unit(conc_unit, time_unit):
    """'mol/dm³' and 's' -> 'mol/(dm³·s)'  (amount per volume per time)."""
    amount, volume = conc_unit.split("/")
    return f"{amount}/({volume}·{time_unit})"


def default_table():
    """Example data (a 1st order reaction with k = 0.1 1/s)."""
    return pd.DataFrame({
        "Time (number)":       [0.0, 1.0, 2.0, 3.0],
        "Time (power of 10)":  [0.0, 1.0, 1.0, 1.0],
        "Conc. (number)":      [1.0, 3.68, 1.35, 4.98],
        "Conc. (power of 10)": [0.0, -1.0, -1.0, -2.0],
    })


# =====================================================================
# DESIGN SETTINGS  (change these to restyle the whole app)
# =====================================================================

PRIMARY = "#0f766e"      # main color (teal)
SECONDARY = "#1d4ed8"    # second color (blue)
ACCENT = "#f97316"       # highlight color (orange)
PALETTE = [PRIMARY, SECONDARY, ACCENT, "#a855f7", "#e11d48", "#64748b"]  # graph colors
AUTHOR = "Made by buldak kalbonara"              # put your name here, for example "Made by Juan"

CSS = """
<style>
.block-container {padding-top: 2rem; max-width: 1150px;}
footer {visibility: hidden;}

/* Top banner */
.hero {
    background: linear-gradient(135deg, __PRIMARY__ 0%, __SECONDARY__ 100%);
    color: white; padding: 1.7rem 1.9rem; border-radius: 20px;
    position: relative; overflow: hidden;
    margin-bottom: 1.2rem; box-shadow: 0 8px 24px rgba(15, 118, 110, 0.25);
}
.hero-title {font-size: 2.2rem; font-weight: 800; line-height: 1.15;}
.hero-sub {font-size: 1.05rem; opacity: 0.93; margin-top: 0.45rem;}
.chips {margin-top: 1rem; display: flex; flex-wrap: wrap; gap: 0.5rem;}
.chips span {
    background: rgba(255,255,255,0.18); border: 1px solid rgba(255,255,255,0.4);
    padding: 0.2rem 0.8rem; border-radius: 999px; font-size: 0.85rem;
}

/* Tabs */
button[data-baseweb="tab"] {font-size: 1.05rem; font-weight: 600; padding: 0.6rem 1.1rem;}
button[data-baseweb="tab"][aria-selected="true"] {color: __PRIMARY__;}
div[data-baseweb="tab-highlight"] {background-color: __PRIMARY__;}

/* Cards and numbers (work in light and dark mode) */
[data-testid="stMetric"] {
    background: rgba(128,128,128,0.08); border: 1px solid rgba(128,128,128,0.25);
    border-left: 6px solid __PRIMARY__; padding: 0.9rem 1.1rem; border-radius: 14px;
}
[data-testid="stVerticalBlockBorderWrapper"] {border-radius: 16px;}
h3 {border-left: 5px solid __ACCENT__; padding-left: 0.6rem;}

/* Buttons */
.stDownloadButton button, .stButton button {
    border-radius: 10px; font-weight: 600; border: 1px solid __PRIMARY__;
}
.stDownloadButton button:hover, .stButton button:hover {
    background: __PRIMARY__; color: white; border-color: __PRIMARY__;
}

/* Graphics */
.stApp {background-image: url("__MOLECULES__"); background-attachment: fixed;}
@keyframes floaty {0%, 100% {transform: translateY(0);} 50% {transform: translateY(-14px);}}
.bubble {position: absolute; border-radius: 50%; background: rgba(255,255,255,0.10);
         animation: floaty 7s ease-in-out infinite; pointer-events: none;}
.hero-inner {position: relative; z-index: 1; display: flex; align-items: center;
             justify-content: space-between; gap: 1.2rem;}
.hero-flask {width: 150px; flex-shrink: 0;}
@media (max-width: 700px) {.hero-flask {display: none;} .hero-title {font-size: 1.7rem;}}
.tabbanner {
    display: flex; align-items: center; gap: 1rem; padding: 0.8rem 1.1rem; margin-bottom: 1rem;
    border-radius: 16px; border: 1px solid rgba(128,128,128,0.2);
    background: linear-gradient(90deg, rgba(15,118,110,0.14), rgba(29,78,216,0.07));
}
.tabbanner img {width: 56px; height: 56px; flex-shrink: 0;}
.tb-title {font-size: 1.25rem; font-weight: 800;}
.tb-text {opacity: 0.8; font-size: 0.95rem;}
.stickers {text-align: center; padding-top: 1rem;}
.stickers span {display: inline-block; font-size: 1.7rem; margin: 0 0.35rem; transition: transform 0.2s;}
.stickers span:hover {transform: rotate(-14deg) scale(1.3);}

.footer-note {text-align: center; opacity: 0.65; font-size: 0.85rem; padding: 1.5rem 0 0.5rem;}
</style>
"""


# ---------------------------------------------------------------------
# GRAPHICS (small drawings written as SVG, so no image files are needed)
# ---------------------------------------------------------------------

# Animated lab flask: waves move and bubbles rise
FLASK_SVG = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 230" width="200" height="230">
<defs><clipPath id="c"><path d="M80 24 H120 V82 L170 172 Q182 200 156 202 H44 Q18 200 30 172 L80 82 Z"/></clipPath></defs>
<g clip-path="url(#c)">
<rect x="0" y="0" width="200" height="230" fill="rgba(255,255,255,0.10)"/>
<path d="M-200 128 q25 -12 50 0 t50 0 t50 0 t50 0 t50 0 t50 0 t50 0 t50 0 V230 H-200 Z" fill="#99f6e4" opacity="0.9">
<animateTransform attributeName="transform" type="translate" from="0 0" to="100 0" dur="3s" repeatCount="indefinite"/></path>
<path d="M-200 140 q25 12 50 0 t50 0 t50 0 t50 0 t50 0 t50 0 t50 0 t50 0 V230 H-200 Z" fill="#2dd4bf" opacity="0.75">
<animateTransform attributeName="transform" type="translate" from="100 0" to="0 0" dur="4s" repeatCount="indefinite"/></path>
<circle cx="75" cy="190" r="6" fill="white" opacity="0.8"><animate attributeName="cy" from="195" to="120" dur="2.6s" repeatCount="indefinite"/><animate attributeName="opacity" from="0.9" to="0" dur="2.6s" repeatCount="indefinite"/></circle>
<circle cx="105" cy="185" r="4" fill="white" opacity="0.8"><animate attributeName="cy" from="195" to="115" dur="2.1s" begin="0.7s" repeatCount="indefinite"/><animate attributeName="opacity" from="0.9" to="0" dur="2.1s" begin="0.7s" repeatCount="indefinite"/></circle>
<circle cx="130" cy="188" r="5" fill="white" opacity="0.8"><animate attributeName="cy" from="195" to="125" dur="3s" begin="1.2s" repeatCount="indefinite"/><animate attributeName="opacity" from="0.9" to="0" dur="3s" begin="1.2s" repeatCount="indefinite"/></circle>
<circle cx="95" cy="176" r="3" fill="white" opacity="0.8"><animate attributeName="cy" from="190" to="110" dur="1.8s" begin="0.3s" repeatCount="indefinite"/><animate attributeName="opacity" from="0.9" to="0" dur="1.8s" begin="0.3s" repeatCount="indefinite"/></circle>
</g>
<path d="M80 24 H120 V82 L170 172 Q182 200 156 202 H44 Q18 200 30 172 L80 82 Z" fill="none" stroke="white" stroke-width="6" stroke-linejoin="round"/>
<line x1="72" y1="24" x2="128" y2="24" stroke="white" stroke-width="7" stroke-linecap="round"/>
<circle cx="100" cy="8" r="4" fill="white" opacity="0.7"><animate attributeName="cy" from="20" to="-6" dur="2.4s" repeatCount="indefinite"/><animate attributeName="opacity" from="0.8" to="0" dur="2.4s" repeatCount="indefinite"/></circle>
</svg>"""

# Faint molecules for the page background (water, benzene ring, CO2)
MOLECULE_TILE = """<svg xmlns="http://www.w3.org/2000/svg" width="260" height="260" viewBox="0 0 260 260">
<g fill="none" stroke="#94a3b8" stroke-width="2" opacity="0.22" stroke-linecap="round">
<polygon points="70,40 94,54 94,82 70,96 46,82 46,54"/>
<circle cx="70" cy="40" r="5"/><circle cx="94" cy="54" r="5"/><circle cx="94" cy="82" r="5"/>
<circle cx="70" cy="96" r="5"/><circle cx="46" cy="82" r="5"/><circle cx="46" cy="54" r="5"/>
<line x1="190" y1="150" x2="168" y2="176"/><line x1="190" y1="150" x2="214" y2="174"/>
<circle cx="190" cy="150" r="11"/><circle cx="165" cy="180" r="7"/><circle cx="217" cy="178" r="7"/>
<line x1="40" y1="200" x2="80" y2="200"/><line x1="80" y1="200" x2="120" y2="200"/>
<circle cx="40" cy="200" r="9"/><circle cx="80" cy="200" r="12"/><circle cx="120" cy="200" r="9"/>
<circle cx="170" cy="40" r="4"/><circle cx="215" cy="95" r="3"/><circle cx="130" cy="120" r="3"/>
</g></svg>"""

# Small icons shown at the top of each tab
ICON_FLASK = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
<path d="M26 6 H38 V24 L54 52 Q58 60 48 60 H16 Q6 60 10 52 L26 24 Z" fill="#ccfbf1" stroke="{PRIMARY}" stroke-width="3.5" stroke-linejoin="round"/>
<path d="M16 42 Q24 37 32 42 T48 42 L54 52 Q58 60 48 60 H16 Q6 60 10 52 Z" fill="{PRIMARY}"/>
<circle cx="26" cy="50" r="2.6" fill="white"><animate attributeName="cy" from="54" to="38" dur="2s" repeatCount="indefinite"/><animate attributeName="opacity" from="1" to="0" dur="2s" repeatCount="indefinite"/></circle>
<circle cx="37" cy="52" r="2" fill="white"><animate attributeName="cy" from="55" to="38" dur="2.6s" begin="0.6s" repeatCount="indefinite"/><animate attributeName="opacity" from="1" to="0" dur="2.6s" begin="0.6s" repeatCount="indefinite"/></circle>
<line x1="22" y1="6" x2="42" y2="6" stroke="{PRIMARY}" stroke-width="4" stroke-linecap="round"/></svg>"""

ICON_CHART = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
<rect x="5" y="5" width="54" height="54" rx="10" fill="none" stroke="{SECONDARY}" stroke-width="3.5"/>
<path d="M13 16 V50 H52" fill="none" stroke="{SECONDARY}" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round"/>
<path d="M16 20 Q26 44 52 46" fill="none" stroke="{ACCENT}" stroke-width="3.5" stroke-linecap="round"/>
<circle cx="19" cy="26" r="3.4" fill="{SECONDARY}"/><circle cx="28" cy="38" r="3.4" fill="{SECONDARY}"/>
<circle cx="38" cy="43" r="3.4" fill="{SECONDARY}"/><circle cx="49" cy="46" r="3.4" fill="{SECONDARY}"/></svg>"""

ICON_FLAME = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
<path d="M32 3 C35 16 51 22 51 41 C51 53 43 61 32 61 C21 61 13 53 13 41 C13 32 18 28 22 21 C24 28 27 29 29 27 C30 21 28 11 32 3 Z" fill="#f97316"/>
<path d="M32 29 C33 38 42 40 42 48 C42 54 38 58 32 58 C26 58 22 54 22 48 C22 41 30 38 32 29 Z" fill="#fde047">
<animate attributeName="opacity" values="1;0.7;1" dur="1.2s" repeatCount="indefinite"/></path></svg>"""


def svg_uri(svg):
    """Turn an SVG drawing into a link that can be used inside a web page."""
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("ascii")


def apply_style():
    """Add the custom colors, cards and tab styling."""
    css = CSS.replace("__PRIMARY__", PRIMARY).replace("__SECONDARY__", SECONDARY)
    css = css.replace("__ACCENT__", ACCENT)
    css = css.replace("__MOLECULES__", svg_uri(MOLECULE_TILE))
    st.markdown(css, unsafe_allow_html=True)


def setup_plots():
    """Give every graph the same clean look."""
    plt.rcParams.update({
        "axes.prop_cycle": matplotlib.cycler(color=PALETTE),
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "lines.linewidth": 2.4,
        "font.size": 11,
        "axes.titleweight": "bold",
        "axes.labelcolor": "#334155",
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def hero():
    """The colorful banner at the top of the page, with an animated flask."""
    bubbles = (
        '<span class="bubble" style="width:120px;height:120px;left:-30px;top:-40px;"></span>'
        '<span class="bubble" style="width:70px;height:70px;left:38%;bottom:-25px;animation-delay:1.5s;"></span>'
        '<span class="bubble" style="width:46px;height:46px;left:62%;top:12px;animation-delay:3s;"></span>'
        '<span class="bubble" style="width:90px;height:90px;right:150px;bottom:-35px;animation-delay:2s;"></span>'
    )
    chips = "".join(f"<span>{t}</span>" for t in [
        "0th · 1st · 2nd order", "Half-life", "Fit your own data",
        "Combustion rates", "Scientific notation"])
    html = (
        '<div class="hero">' + bubbles +
        '<div class="hero-inner"><div>'
        '<div class="hero-title">🧪 Reaction Kinetics Lab</div>'
        '<div class="hero-sub">Simulate reactions, fit your lab data, and calculate '
        'reaction rates, all in your browser.</div>'
        f'<div class="chips">{chips}</div></div>'
        f'<img class="hero-flask" src="{svg_uri(FLASK_SVG)}" alt="Animated lab flask"/>'
        '</div></div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def tab_banner(icon_svg, title, text):
    """A small header with an icon at the top of each tab."""
    st.markdown(
        f'<div class="tabbanner"><img src="{svg_uri(icon_svg)}" alt=""/>'
        f'<div><div class="tb-title">{title}</div><div class="tb-text">{text}</div></div></div>',
        unsafe_allow_html=True)


def how_to_use():
    """A short guide that opens and closes."""
    with st.expander("ℹ️ How to use this app"):
        st.markdown(
            """
**1. Pick your units** at the top (they are used in every tab).

**🧪 Simulator:** enter k and C0 to compare 0th, 1st and 2nd order reactions
and see their half-lives.

**📈 Fit my data:** type your experiment results. The app finds the best reaction
order and the rate constant k.

**🔥 Reaction rates:** type a reaction like `A + B -> C`, or pick combustion and
type a fuel like `C3H8`. Choose elementary or complex, add your mixtures, and get the
rate of consumption and formation of every species.

**Tip:** you can type scientific notation as `2.5e-3` or `2.5*10^-3`.
            """
        )


def footer():
    text = "Built with Python and Streamlit"
    if AUTHOR.strip():
        text = f"{AUTHOR.strip()} · " + text
    st.markdown(
        '<div class="stickers"><span>⚗️</span><span>🔬</span><span>🧬</span>'
        '<span>🧫</span><span>🔥</span><span>📈</span></div>'
        f'<div class="footer-note">{text}</div>',
        unsafe_allow_html=True)


# =====================================================================
# TAB 1: SIMULATOR
# =====================================================================

def simulator_tab(conc_unit, time_unit):
    left, right = st.columns([5, 7], gap="large")

    with left:
        with st.container(border=True):
            st.subheader("⚙️ Inputs")
            st.caption("Enter a rate constant and a starting concentration "
                       "to compare the three reaction orders.")
            k = sci_input("Rate constant k", 1.0, -1, key="sim_k")
            C0 = sci_input(f"Initial concentration C0 ({conc_unit})", 1.0, 0, key="sim_c0")
            t_max = st.slider(f"Time to simulate ({time_unit})", 1, 1000, 50)

    if k <= 0 or C0 <= 0:
        right.error("k and C0 must both be greater than zero.")
        return

    t = np.linspace(0, t_max, 300)
    curves = concentration_curves(k, C0, t)

    with right:
        with st.container(border=True):
            st.subheader("📉 Concentration vs time")
            fig, ax = plt.subplots()
            for order, C in curves.items():
                ax.plot(t, C, label=f"{order}{['th', 'st', 'nd'][order]} order")
            ax.set_xlabel(f"Time ({time_unit})")
            ax.set_ylabel(f"Concentration ({conc_unit})")
            ax.legend()
            st.pyplot(fig)
            download_buttons(fig, "kinetics_simulation", key="sim")

    st.subheader("⏱️ Half-life")
    lives = half_lives(k, C0)
    c0, c1, c2 = st.columns(3)
    c0.metric("0th order", f"{lives[0]:.4g} {time_unit}")
    c1.metric("1st order", f"{lives[1]:.4g} {time_unit}")
    c2.metric("2nd order", f"{lives[2]:.4g} {time_unit}")
    st.caption("0th: C0/(2k)   |   1st: ln(2)/k   |   2nd: 1/(k·C0)")


# =====================================================================
# TAB 2: FIT MY OWN DATA
# =====================================================================

def fit_tab(conc_unit, time_unit):
    st.write("Type your experimental data. Every value is "
             "**number × 10^(power of 10)**. Example: number 4.98 and power "
             "-2 means 4.98 × 10⁻². Add rows with the + at the bottom of the "
             "table; delete rows by selecting them and pressing Delete.")

    table = st.data_editor(default_table(), num_rows="dynamic",
                           key="data_table")

    # Keep only complete numeric rows
    clean = table.apply(pd.to_numeric, errors="coerce").dropna()
    if len(clean) < 3:
        st.warning("Please enter at least 3 complete data points.")
        return

    t = sci_value(clean["Time (number)"],
                  clean["Time (power of 10)"]).to_numpy(dtype=float)
    C = sci_value(clean["Conc. (number)"],
                  clean["Conc. (power of 10)"]).to_numpy(dtype=float)

    # Sort by time so the graph looks right
    order_idx = np.argsort(t)
    t, C = t[order_idx], C[order_idx]

    st.write("**Your values after converting the scientific notation:**")
    st.dataframe(pd.DataFrame({f"Time ({time_unit})": t,
                               f"Concentration ({conc_unit})": C}))

    # Check the input before fitting
    if np.any(C <= 0):
        st.error("Every concentration must be greater than zero.")
        return
    if np.unique(t).size < 3:
        st.error("Please use at least 3 different time values.")
        return

    # Fit all three orders
    fits = [fit_order(order, t, C) for order in (0, 1, 2)]
    results = pd.DataFrame({
        "Order": [f["order"] for f in fits],
        "k": [f"{f['k']:.4g} {k_units(f['order'], time_unit, conc_unit)}"
              for f in fits],
        "Slope": [f"{f['slope']:.4g}" for f in fits],
        "R²": [f"{f['r2']:.4f}" for f in fits],
    })
    st.subheader("📊 Results")
    st.dataframe(results, hide_index=True)

    # Best fit = highest R², but k must be positive to make sense
    valid = [f for f in fits if f["k"] > 0]
    if not valid:
        st.error("No order gives a positive k. Please check your data.")
        return
    best = max(valid, key=lambda f: f["r2"])
    units = k_units(best["order"], time_unit, conc_unit)
    st.success(f"Best fit: order {best['order']}  |  "
               f"k = {best['k']:.4g} {units}  |  R² = {best['r2']:.4f}")

    # Graphs: data + fitted curve, and the straight-line plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    tt = np.linspace(t.min(), t.max(), 200)
    ax1.scatter(t, C, color="black", label="Your data", zorder=3)
    ax1.plot(tt, model_curve(best["order"], best["slope"],
                             best["intercept"], tt),
             label=f"Order {best['order']} fit")
    ax1.set_xlabel(f"Time ({time_unit})")
    ax1.set_ylabel(f"Concentration ({conc_unit})")
    ax1.legend()

    y_labels = {0: "C", 1: "ln(C)", 2: "1/C"}
    ax2.scatter(t, linearize(best["order"], C), color="black", zorder=3)
    ax2.plot(tt, best["slope"] * tt + best["intercept"])
    ax2.set_xlabel(f"Time ({time_unit})")
    ax2.set_ylabel(y_labels[best["order"]])
    ax2.set_title("Straight-line check")

    fig.tight_layout()
    st.pyplot(fig)
    download_buttons(fig, "kinetics_fit", key="fit")


# =====================================================================
# TAB 3: REACTION RATES  (A + B -> C, combustion, elementary / complex)
# =====================================================================

ELEMENTS = set("""H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn
Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te
I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au
Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu""".split())


def parse_number(text):
    """Read 2.5, 2.5e-3, 2.5*10^-3, 2.5x10^-3 or 10^-3. Returns None if invalid."""
    s = str(text).strip().lower().replace(" ", "").replace("×", "x").replace("**", "^")
    m = re.fullmatch(r"10\^\(?([-+]?\d+)\)?", s)
    if m:
        return 10.0 ** int(m.group(1))
    m = re.fullmatch(r"([-+]?(?:\d+\.?\d*|\.\d+)(?:e[-+]?\d+)?)"
                     r"(?:[x*]10\^\(?([-+]?\d+)\)?)?", s)
    if not m:
        return None
    value = float(m.group(1))
    if m.group(2) is not None:
        value *= 10.0 ** int(m.group(2))
    return value


def parse_formula(formula):
    """'C3H8' -> {'C': 3, 'H': 8}. Brackets such as Ca(OH)2 also work."""
    stack = [{}]
    i = 0
    while i < len(formula):
        ch = formula[i]
        if ch == "(":
            stack.append({})
            i += 1
        elif ch == ")":
            if len(stack) == 1:
                raise ValueError(f"Brackets do not match in '{formula}'.")
            group = stack.pop()
            i += 1
            num = re.match(r"\d+", formula[i:])
            mult = int(num.group()) if num else 1
            if num:
                i += len(num.group())
            for el, n in group.items():
                stack[-1][el] = stack[-1].get(el, 0) + n * mult
        else:
            m = re.match(r"([A-Z][a-z]?)(\d*)", formula[i:])
            if not m:
                raise ValueError(f"Cannot read the formula '{formula}'.")
            n = int(m.group(2)) if m.group(2) else 1
            stack[-1][m.group(1)] = stack[-1].get(m.group(1), 0) + n
            i += len(m.group(0))
    if len(stack) != 1:
        raise ValueError(f"Brackets do not match in '{formula}'.")
    return stack[0]


def parse_reaction(text):
    """'2 H2 + O2 -> 2 H2O' -> (reactants, products), each a list of (coef, formula)."""
    s = text.replace("→", "->").replace("-->", "->").replace("=>", "->")
    s = s.replace("=", "->") if "->" not in s else s
    parts = s.split("->")
    if len(parts) != 2:
        raise ValueError("Write the reaction with one arrow, for example: A + B -> C")
    sides = []
    for side in parts:
        terms = []
        for term in side.split("+"):
            term = term.strip()
            if not term:
                raise ValueError("A term is empty. Check the + signs.")
            m = re.fullmatch(r"(\d+)?\s*([A-Za-z(][A-Za-z0-9()]*)", term)
            if not m:
                raise ValueError(f"Cannot read '{term}'.")
            coef = int(m.group(1)) if m.group(1) else 1
            terms.append((coef, m.group(2)))
        sides.append(terms)
    return sides[0], sides[1]


def nullspace(rows, ncols):
    """Exact null space of a matrix made of Fractions (Gauss-Jordan)."""
    m = [r[:] for r in rows]
    pivots, r = [], 0
    for c in range(ncols):
        pr = next((i for i in range(r, len(m)) if m[i][c] != 0), None)
        if pr is None:
            continue
        m[r], m[pr] = m[pr], m[r]
        m[r] = [x / m[r][c] for x in m[r]]
        for i in range(len(m)):
            if i != r and m[i][c] != 0:
                f = m[i][c]
                m[i] = [a - f * b for a, b in zip(m[i], m[r])]
        pivots.append(c)
        r += 1
        if r == len(m):
            break
    vectors = []
    for fcol in [c for c in range(ncols) if c not in pivots]:
        v = [Fraction(0)] * ncols
        v[fcol] = Fraction(1)
        for i, pc in enumerate(pivots):
            v[pc] = -m[i][fcol]
        vectors.append(v)
    return vectors


def balance(reactants, products):
    """Smallest whole-number coefficients that balance the reaction."""
    species = [f for _, f in reactants] + [f for _, f in products]
    comps = [parse_formula(f) for f in species]
    elements = sorted({el for c in comps for el in c})
    if not all(el in ELEMENTS for el in elements):
        raise ValueError("Auto-balance needs real chemical formulas (like C3H8). "
                         "For letters such as A + B -> C, untick it and type the coefficients.")
    n_react = len(reactants)
    rows = [[Fraction(c.get(el, 0)) * (1 if j < n_react else -1)
             for j, c in enumerate(comps)] for el in elements]
    vectors = nullspace(rows, len(species))
    if len(vectors) != 1:
        raise ValueError("This reaction cannot be balanced in one unique way. "
                         "Check the species (every element must appear on both sides).")
    v = vectors[0]
    if all(x <= 0 for x in v):
        v = [-x for x in v]
    if any(x <= 0 for x in v):
        raise ValueError("These species cannot be balanced. Check the formulas.")
    den = 1
    for x in v:
        den = den * x.denominator // gcd(den, x.denominator)
    ints = [int(x * den) for x in v]
    g = 0
    for x in ints:
        g = gcd(g, x)
    return [x // g for x in ints]


def balance_problem(species, coefs, n_react):
    """For real formulas, return a warning text if the typed coefficients do not balance."""
    comps = [parse_formula(f) for f in species]
    elements = {el for c in comps for el in c}
    if not all(el in ELEMENTS for el in elements):
        return None                      # letters like A, B: nothing to check
    for el in sorted(elements):
        left = sum(coefs[j] * comps[j].get(el, 0) for j in range(n_react))
        right = sum(coefs[j] * comps[j].get(el, 0) for j in range(n_react, len(species)))
        if left != right:
            return (f"This reaction is not balanced: {el} has {left} atoms on the left "
                    f"and {right} on the right. Fix the coefficients or tick 'Balance automatically'.")
    return None


def reaction_string(coefs, species, n_react):
    """Nice text such as: C3H8 + 5 O2 → 3 CO2 + 4 H2O."""
    def side(items):
        return " + ".join((f"{c} " if c != 1 else "") + s for c, s in items)
    left = side(list(zip(coefs[:n_react], species[:n_react])))
    right = side(list(zip(coefs[n_react:], species[n_react:])))
    return f"{left} → {right}"


def reaction_rate(k, conc, orders, reactant_idx):
    """Rate law: r = k * product(C_i ^ order_i). It is zero if a reactant is used up."""
    if np.any(conc[reactant_idx] <= 0):
        return 0.0
    r = k
    for i, n in zip(reactant_idx, orders):
        r *= conc[i] ** n
    return r


def simulate(k, conc0, nu, orders, reactant_idx, t_end, steps=2000):
    """Batch reactor: dC_i/dt = nu_i * r, solved with the Runge-Kutta 4 method."""
    t = np.linspace(0, t_end, steps + 1)
    h = t[1] - t[0]
    C = np.zeros((steps + 1, len(conc0)))
    C[0] = conc0

    def dcdt(c):
        return nu * reaction_rate(k, c, orders, reactant_idx)

    for i in range(steps):
        c = C[i]
        s1 = dcdt(c)
        s2 = dcdt(c + h / 2 * s1)
        s3 = dcdt(c + h / 2 * s2)
        s4 = dcdt(c + h * s3)
        C[i + 1] = np.maximum(c + h / 6 * (s1 + 2 * s2 + 2 * s3 + s4), 0)
    rates = np.array([reaction_rate(k, c, orders, reactant_idx) for c in C])
    return t, C, rates


def rates_tab(conc_unit, time_unit):
    st.write("Type a reaction, choose the rate law, and get the reaction rate plus the "
             "rate of consumption of each reactant and rate of formation of each product, "
             "for every mixture you enter.")

    # ---- 1. The reaction -------------------------------------------------
    mode = st.radio("Reaction type", ["Type my own reaction", "Combustion of a fuel"],
                    horizontal=True)
    if mode == "Combustion of a fuel":
        fuel = st.text_input("Fuel formula (example: CH4, C3H8, C2H5OH → write C2H6O)",
                             "C3H8").strip()
        text = f"{fuel} + O2 -> CO2 + H2O"
        auto_balance = True
    else:
        text = st.text_input("Reaction (use -> for the arrow)", "A + B -> C")
        auto_balance = st.checkbox("Balance automatically (real chemical formulas only)",
                                   value=False)

    try:
        reactants, products = parse_reaction(text)
        species = [f for _, f in reactants] + [f for _, f in products]
        if len(set(species)) != len(species):
            raise ValueError("Each species can appear only once in the reaction.")
        n_react = len(reactants)
        if auto_balance:
            coefs = balance(reactants, products)
        else:
            coefs = [c for c, _ in reactants] + [c for c, _ in products]
            problem = balance_problem(species, coefs, n_react)
            if problem:
                st.warning(problem)
    except ValueError as err:
        st.error(str(err))
        return

    st.success("Reaction used:  " + reaction_string(coefs, species, n_react))
    nu = np.array([-c for c in coefs[:n_react]] + [c for c in coefs[n_react:]], dtype=float)
    reactant_idx = list(range(n_react))
    key_text = "_".join(species + [str(c) for c in coefs])

    # ---- 2. The rate law -------------------------------------------------
    st.subheader("📐 Rate law")
    law = st.radio("Reaction kind", ["Elementary", "Complex"], horizontal=True)
    if law == "Elementary":
        orders = [float(c) for c in coefs[:n_react]]
        st.caption("Elementary: the order of each reactant equals its coefficient.")
        if mode == "Combustion of a fuel":
            st.info("Overall combustion reactions are almost never elementary. "
                    "Choose 'Complex' and enter the orders found from experiments.")
    else:
        st.caption("Complex (many steps): the orders come from experiments. They can be "
                   "fractions, zero, or negative. Edit the Order column.")
        order_df = st.data_editor(
            pd.DataFrame({"Reactant": species[:n_react],
                          "Order": [float(c) for c in coefs[:n_react]]}),
            disabled=["Reactant"], hide_index=True, key="orders_" + key_text)
        order_values = pd.to_numeric(order_df["Order"], errors="coerce")
        if order_values.isna().any():
            st.error("Every order must be a number.")
            return
        orders = order_values.tolist()

    n_total = sum(orders)
    law_text = " ".join(f"[{s}]^{n:g}" for s, n in zip(species[:n_react], orders))
    st.write(f"**r = k {law_text}**   (overall order = {n_total:g})")

    # ---- 3. The rate constant ---------------------------------------------
    k_mode = st.radio("Rate constant", ["Enter k", "Use the Arrhenius equation"],
                      horizontal=True)
    if k_mode == "Enter k":
        k = sci_input("Rate constant k", 1.0, -2, key="rate_k")
    else:
        A = sci_input("Pre-exponential factor A", 1.0, 6, key="rate_A")
        Ea = st.number_input("Activation energy Ea (kJ/mol)", value=50.0, min_value=0.0)
        T = st.number_input("Temperature T (K)", value=500.0, min_value=1.0)
        k = A * np.exp(-Ea * 1000 / (8.314 * T))
        st.write(f"k = A·exp(−Ea / RT) = **{k:.4g}**")
    st.caption(f"Units of k: ({conc_unit})^{1 - n_total:g} · {time_unit}⁻¹ "
               f"(for a 1st-order reaction this is just 1/{time_unit}). "
               "Enter k (or A) in these units.")
    if k <= 0:
        st.error("k must be greater than zero.")
        return

    # ---- 4. The mixtures --------------------------------------------------
    st.subheader("🧫 Mixtures")
    st.write(f"Each row is one mixture. Concentrations are in {conc_unit}. Scientific notation "
             "works: 2.5e-3 or 2.5*10^-3. Add rows with the + at the bottom.")
    default_rows = []
    for name, scale in (("Mixture 1", 1.0), ("Mixture 2", 2.0)):
        row = {"Mixture": name}
        for i, s in enumerate(species):
            row[s] = str(scale) if i < n_react else "0"
        default_rows.append(row)
    mix = st.data_editor(pd.DataFrame(default_rows), num_rows="dynamic",
                         key="mix_" + key_text)

    names, conc_list = [], []
    for idx, row in mix.reset_index(drop=True).iterrows():
        if all(pd.isna(row[s]) for s in species):
            continue                                   # empty row: skip it
        conc = []
        for s in species:
            value = parse_number(row[s]) if pd.notna(row[s]) else None
            if value is None or value < 0:
                st.error(f"Row {idx + 1}: '{row[s]}' is not a valid concentration for {s}.")
                return
            conc.append(value)
        label = row["Mixture"]
        names.append(str(label) if pd.notna(label) and str(label).strip()
                     else f"Mixture {idx + 1}")
        conc_list.append(np.array(conc))
    if not conc_list:
        st.warning("Add at least one mixture.")
        return

    # ---- 5. Rates for every mixture -----------------------------------------
    table_rows = []
    for name, conc in zip(names, conc_list):
        r = reaction_rate(k, conc, orders, reactant_idx)
        row = {"Mixture": name, "Reaction rate r": f"{r:.4g}"}
        for i, s in enumerate(species):
            if i < n_react:
                row[f"Consumption of {s}"] = f"{-nu[i] * r:.4g}"
            else:
                row[f"Formation of {s}"] = f"{nu[i] * r:.4g}"
        table_rows.append(row)
    st.subheader("⚡ Rates at the starting concentrations")
    st.dataframe(pd.DataFrame(table_rows), hide_index=True)
    st.caption(f"All rates are in {rate_unit(conc_unit, time_unit)}.  Reaction rate r = rate of consumption of a "
               "reactant ÷ its coefficient = rate of formation of a product ÷ its coefficient.")

    # ---- 6. Concentration vs time ---------------------------------------------
    st.subheader("📉 Concentration vs time (batch reactor)")
    choice = st.selectbox("Mixture to simulate", names)
    t_end = sci_input(f"Simulation time ({time_unit})", 1.0, 1, key="rate_tend")
    if t_end <= 0:
        st.error("Simulation time must be greater than zero.")
        return
    conc0 = conc_list[names.index(choice)]
    t, C, r_t = simulate(k, conc0, nu, orders, reactant_idx, t_end)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for i, s in enumerate(species):
        ax1.plot(t, C[:, i], label=s)
    ax1.set_xlabel(f"Time ({time_unit})")
    ax1.set_ylabel(f"Concentration ({conc_unit})")
    ax1.legend()
    ax2.plot(t, r_t)
    ax2.set_xlabel(f"Time ({time_unit})")
    ax2.set_ylabel(f"Reaction rate r ({rate_unit(conc_unit, time_unit)})")
    fig.tight_layout()
    st.pyplot(fig)
    download_buttons(fig, "reaction_rates", key="rates")


# =====================================================================
# MAIN
# =====================================================================

def main():
    st.set_page_config(page_title="Reaction Kinetics Lab", page_icon="🧪", layout="wide")
    apply_style()
    setup_plots()
    hero()

    # Units are chosen once and used in every tab.
    col1, col2 = st.columns(2)
    conc_unit = col1.selectbox("Concentration unit",
                               ["mol/L", "mol/dm³", "mmol/L", "mol/m³", "kmol/m³"])
    time_unit = col2.selectbox("Time unit", ["s", "min", "h"])
    st.caption("Note: 1 dm³ = 1 L, so mol/dm³ is the same as mol/L. The units are labels: "
               "the app does not convert your numbers, so enter k and your data in the "
               "units you choose here.")

    tab1, tab2, tab3 = st.tabs(["🧪 Simulator", "📈 Fit my data", "🔥 Reaction rates"])
    with tab1:
        tab_banner(ICON_FLASK, "Simulator", "Compare 0th, 1st and 2nd order reactions side by side.")
        simulator_tab(conc_unit, time_unit)
    with tab2:
        tab_banner(ICON_CHART, "Fit my data", "Type your lab results and find the best reaction order and k.")
        fit_tab(conc_unit, time_unit)
    with tab3:
        tab_banner(ICON_FLAME, "Reaction rates", "Rates of consumption and formation for reactions and combustion.")
        rates_tab(conc_unit, time_unit)

    how_to_use()
    footer()


if __name__ == "__main__":
    main()
