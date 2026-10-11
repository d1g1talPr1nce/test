#!/usr/bin/env python3
"""
Digitální degradace chaosu u spojitých systémů (Lorenz, Rössler) – 3D animace.

Modrá dráha   = "přesná" trajektorie (float64, RK4)
Červená dráha = stejný systém, ale stav se po KAŽDÉM kroku uloží na hrubou mřížku
                (fixed-point s krokem --grid). Simuluje to omezenou přesnost
                paměti: konečně mnoho stavů -> Dirichlet -> nutně cyklus.

Ovládání: myš = otáčení/zoom, mezerník = pauza, R = auto-rotace zap/vyp
Příklady:
    python chaos_ode_degradace.py                          # Lorenz, grid 0.05
    python chaos_ode_degradace.py --grid 0.1               # hrubší -> kratší cyklus
    python chaos_ode_degradace.py --system rossler --grid 0.05
    python chaos_ode_degradace.py --save-png out.png       # bez okna, jen konečný stav
"""
import argparse
import numpy as np

SYSTEMS = {
    # název: (pravá strana, parametry, počáteční stav, dt, výchozí mřížka)
    "lorenz": (lambda s, p: (p[0]*(s[1]-s[0]),
                             s[0]*(p[1]-s[2]) - s[1],
                             s[0]*s[1] - p[2]*s[2]),
               (10.0, 28.0, 8.0/3.0), (1.0, 1.0, 1.0), 0.01, 0.05),
    "rossler": (lambda s, p: (-s[1]-s[2],
                              s[0] + p[0]*s[1],
                              p[1] + s[2]*(s[0]-p[2])),
                (0.2, 0.2, 5.7), (1.0, 1.0, 0.0), 0.02, 0.01),
}


def rk4(f, s, dt, p):
    add = lambda a, b, h: tuple(u + h*v for u, v in zip(a, b))
    k1 = f(s, p)
    k2 = f(add(s, k1, dt/2), p)
    k3 = f(add(s, k2, dt/2), p)
    k4 = f(add(s, k3, dt), p)
    return tuple(u + dt/6*(a + 2*b + 2*c + d)
                 for u, a, b, c, d in zip(s, k1, k2, k3, k4))


def exact_orbit(f, s0, dt, p, n):
    out = np.empty((n, 3)); s = s0
    for i in range(n):
        out[i] = s
        s = rk4(f, s, dt, p)
    return out


def quantized_orbit(f, s0, dt, p, n, q):
    """Po každém kroku se stav zaokrouhlí na celočíselnou mřížku (násobky q).
    Stav = trojice celých čísel -> přesná detekce opakování (holubníkový princip)."""
    out = np.empty((n, 3))
    s = tuple(int(round(v/q)) for v in s0)
    seen, c_start, period = {}, None, None
    for i in range(n):
        real = tuple(v*q for v in s)
        out[i] = real
        if s in seen:
            c_start, period = seen[s], i - seen[s]
            out = out[:i + 1]          # dál už se jen opakuje
            break
        seen[s] = i
        nxt = rk4(f, real, dt, p)
        if not all(abs(v) < 1e6 for v in nxt):   # kvantizace může systém rozkmitat
            raise SystemExit("Kvantovaná dráha divergovala – zkus jemnější --grid.")
        s = tuple(int(round(v/q)) for v in nxt)
    return out, c_start, period


def main():
    ap = argparse.ArgumentParser()
    
    ap.add_argument("--system", choices=SYSTEMS, default="lorenz")
    ap.add_argument("--grid", type=float, default=None, help="krok mřížky (menší = přesnější)")
    ap.add_argument("--steps", type=int, default=6000, help="minimum vykreslených kroků")
    ap.add_argument("--max-steps", type=int, default=400000)
    ap.add_argument("--per-frame", type=int, default=10)
    ap.add_argument("--interval", type=int, default=40)
    ap.add_argument("--save-png", type=str, default=None)
    args = ap.parse_args()

    import matplotlib
    if args.save_png:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    matplotlib.rcParams['agg.path.chunksize'] = 10000

    f, p, s0, dt, gdef = SYSTEMS[args.system]
    q = args.grid or gdef

    lo, c_start, period = quantized_orbit(f, s0, dt, p, args.max_steps, q)
    if c_start is None:
        print("Cyklus nenalezen – zvětši --grid nebo --max-steps.")
    else:
        print(f"[{args.system}, grid={q}] vstup do cyklu v kroku {c_start}, perioda = {period}"
              + ("  (pevný bod – stagnace)" if period == 1 else ""))
    M = len(lo)
    n_show = max(M + 300, args.steps) if c_start is not None else M
    n_show = min(n_show, args.max_steps)
    hi = exact_orbit(f, s0, dt, p, n_show)
    if len(lo) < n_show:       # po vstupu do cyklu pokračuje smyčka dál
        cyc = lo[c_start:c_start + period]
        extra = np.resize(cyc, (n_show - len(lo), 3)) if period else lo[-1:]
        lo = np.vstack([lo, extra])
    M = n_show
    frames = int(np.ceil(M / args.per_frame))

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    allp = np.vstack([hi, lo])
    ax.set_xlim(allp[:, 0].min(), allp[:, 0].max())
    ax.set_ylim(allp[:, 1].min(), allp[:, 1].max())
    ax.set_zlim(allp[:, 2].min(), allp[:, 2].max())
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title(f"{args.system.capitalize()}: float64 vs mřížka {q}")

    (l_hi,) = ax.plot([], [], [], color="tab:blue", lw=0.6, alpha=0.7, label="float64 (chaos)")
    (l_lo,) = ax.plot([], [], [], color="tab:red", lw=0.9, alpha=0.8, label="omezená přesnost")
    (l_cy,) = ax.plot([], [], [], color="gold", lw=2.5, label="cyklus (smyčka)")
    (p_hi,) = ax.plot([], [], [], "o", color="navy", ms=6)
    (p_lo,) = ax.plot([], [], [], "o", color="darkred", ms=6)
    ax.legend(loc="upper left")
    status = ax.text2D(0.02, 0.02, "", transform=ax.transAxes)
    state = {"rotate": True, "run": True}

    def on_key(ev):
        if ev.key == "r":
            state["rotate"] = not state["rotate"]
        elif ev.key == " ":
            (ani.pause if state["run"] else ani.resume)()
            state["run"] = not state["run"]
    fig.canvas.mpl_connect("key_press_event", on_key)

    def update(fr):
        k = min((fr + 1) * args.per_frame, M)
        l_hi.set_data_3d(*hi[:k].T)
        l_lo.set_data_3d(*lo[:k].T)
        p_hi.set_data_3d(*[[c] for c in hi[k-1]])
        p_lo.set_data_3d(*[[c] for c in lo[k-1]])
        txt = f"krok n = {k}  (t = {k*dt:.1f})"
        if c_start is not None and k >= c_start + period:
            l_cy.set_data_3d(*lo[c_start:c_start + period + 1].T)
            txt += f"   <- uvízl v cyklu, perioda {period}"
        status.set_text(txt)
        if state["rotate"]:
            ax.view_init(elev=20 + 8*np.sin(fr/80), azim=30 + fr*0.4)
        return l_hi, l_lo, l_cy, p_hi, p_lo, status

    ani = FuncAnimation(fig, update, frames=frames, interval=args.interval,
                        blit=False, repeat=False)
    if args.save_png:
        update(frames - 1)
        fig.savefig(args.save_png, dpi=110)
    else:
        plt.show()


if __name__ == "__main__":
    main()