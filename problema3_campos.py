"""
Problema 3 — Campos vectoriales y gradientes
Cálculo Vectorial · Universidad de La Salle · 2026-2

Genera las gráficas de la exposición:
  1) F = ∇f, f(x,y) = x²y − y³, con curvas de nivel de f, la curva f = 0
     y el vector F(1,1) perpendicular a la rama y = x.
  2) G(x,y) = <−y, x> (campo rotacional) con circunferencias de referencia.
  3) Comparación de circulación: ∮ F·dr = 0 vs ∮ G·dr = 2π en x² + y² = 1.

Ejecutar:  python problema3_campos.py   (requiere numpy y matplotlib)
"""
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# Definiciones
# --------------------------------------------------------------------------
def f(x, y):
    return x**2 * y - y**3

def F(x, y):                      # ∇f = <2xy, x² − 3y²>
    return 2 * x * y, x**2 - 3 * y**2

def G(x, y):                      # campo rotacional
    return -y, x

AZUL, NARANJA, GRIS, TINTA = "#1f5aa6", "#d9661f", "#9aa3ad", "#16202a"
plt.rcParams.update({"font.size": 12, "axes.titlesize": 14,
                     "axes.spines.top": False, "axes.spines.right": False})

# --------------------------------------------------------------------------
# Figura 1: F = ∇f en [−2,2]×[−2,2]
# --------------------------------------------------------------------------
x = np.linspace(-2, 2, 17)
X, Y = np.meshgrid(x, x)
U, V = F(X, Y)
M = np.hypot(U, V)
M[M == 0] = 1

xs = np.linspace(-2, 2, 400)
XS, YS = np.meshgrid(xs, xs)

fig, ax = plt.subplots(figsize=(7, 7))
cs = ax.contour(XS, YS, f(XS, YS), levels=np.arange(-6, 7, 1.5),
                colors=GRIS, linewidths=0.8)
ax.clabel(cs, fontsize=8, fmt="%.1f")
# curva de nivel f = 0:  y(x − y)(x + y) = 0  → y = 0, y = x, y = −x
t = np.linspace(-2, 2, 2)
for yy, lbl in [(0 * t, "f = 0"), (t, None), (-t, None)]:
    ax.plot(t, yy, color=TINTA, lw=2, label=lbl)
# campo normalizado (dirección) coloreado por magnitud
q = ax.quiver(X, Y, U / M, V / M, M, cmap="Blues", scale=26, width=0.004)
fig.colorbar(q, ax=ax, shrink=0.75, label="|F|")
# F(1,1) = <2, −2> y tangente a y = x
ax.plot(1, 1, "o", color=NARANJA, ms=8, zorder=5)
ax.annotate("", xy=(1 + 0.5, 1 - 0.5), xytext=(1, 1),
            arrowprops=dict(arrowstyle="-|>", color=NARANJA, lw=2.5))
ax.annotate("", xy=(1.5, 1.5), xytext=(0.5, 0.5),
            arrowprops=dict(arrowstyle="<|-|>", color=AZUL, lw=2))
ax.text(1.95, 0.30, "F(1,1) = ⟨2, −2⟩", color=NARANJA, fontsize=12, ha="right",
        bbox=dict(fc="white", ec="none", alpha=0.85))
ax.text(1.95, 1.72, "tangente ⟨1, 1⟩", color=AZUL, fontsize=11, ha="right",
        bbox=dict(fc="white", ec="none", alpha=0.85))
ax.set(xlim=(-2, 2), ylim=(-2, 2), aspect="equal", xlabel="x", ylabel="y",
       title="F = ∇f = ⟨2xy, x² − 3y²⟩  y curvas de nivel de f\nDominio: [−2, 2] × [−2, 2]")
ax.legend(loc="lower left")
fig.tight_layout()
fig.savefig("fig1_gradiente_F.png", dpi=180)

# --------------------------------------------------------------------------
# Figura 2: G = <−y, x>
# --------------------------------------------------------------------------
U2, V2 = G(X, Y)
M2 = np.hypot(U2, V2)
fig, ax = plt.subplots(figsize=(7, 7))
th = np.linspace(0, 2 * np.pi, 300)
for r in (0.5, 1, 1.5, 2):
    ax.plot(r * np.cos(th), r * np.sin(th), color=GRIS, lw=0.8, ls="--")
q = ax.quiver(X, Y, U2, V2, M2, cmap="Oranges", scale=40, width=0.004)
fig.colorbar(q, ax=ax, shrink=0.75, label="|G| = r")
ax.plot(0, 0, "o", color=TINTA)
ax.set(xlim=(-2, 2), ylim=(-2, 2), aspect="equal", xlabel="x",
       ylabel="y", title="G = ⟨−y, x⟩: rotación antihoraria, |G| = r\nDominio: [−2, 2] × [−2, 2]")
fig.tight_layout()
fig.savefig("fig2_campo_G.png", dpi=180)

# --------------------------------------------------------------------------
# Comprobación numérica de las circulaciones en x² + y² = 1
# --------------------------------------------------------------------------
tt = np.linspace(0, 2 * np.pi, 20001)
xc, yc = np.cos(tt), np.sin(tt)
dx, dy = -np.sin(tt), np.cos(tt)
Fx, Fy = F(xc, yc)
Gx, Gy = G(xc, yc)
circ_F = round(np.trapezoid(Fx * dx + Fy * dy, tt), 10) + 0.0
circ_G = np.trapezoid(Gx * dx + Gy * dy, tt)
print(f"∮ F·dr = {circ_F:.6f}   (esperado 0)")
print(f"∮ G·dr = {circ_G:.6f}   (esperado 2π = {2*np.pi:.6f})")
print("F(1,1) =", F(1, 1), "   F(1,1)·<1,1> =", np.dot(F(1, 1), (1, 1)))

# Figura 3: comparación lado a lado con la circunferencia unitaria
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
for ax, (campo, nombre, color, val) in zip(
        axs, [(F, "F = ∇f (conservativo)", AZUL, circ_F),
              (G, "G = ⟨−y, x⟩ (no conservativo)", NARANJA, circ_G)]):
    xg = np.linspace(-1.6, 1.6, 13)
    XG, YG = np.meshgrid(xg, xg)
    Ug, Vg = campo(XG, YG)
    Mg = np.hypot(Ug, Vg); Mg[Mg == 0] = 1
    ax.quiver(XG, YG, Ug / Mg, Vg / Mg, color=color, alpha=0.7, scale=22)
    ax.plot(np.cos(th), np.sin(th), color=TINTA, lw=2.5)
    ax.annotate("", xy=(np.cos(0.9), np.sin(0.9)),
                xytext=(np.cos(0.75), np.sin(0.75)),
                arrowprops=dict(arrowstyle="-|>", color=TINTA, lw=2.5))
    ax.set(aspect="equal", xlim=(-1.7, 1.7), ylim=(-1.7, 1.7),
           title=f"{nombre}\n∮ = {val:.4f}")
fig.tight_layout()
fig.savefig("fig3_circulacion.png", dpi=180)
print("Figuras guardadas: fig1_gradiente_F.png, fig2_campo_G.png, fig3_circulacion.png")
