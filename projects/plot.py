# universal_plot.py
from __future__ import annotations
import os
from typing import Optional, Tuple, Dict, Iterable, Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize

Array = np.ndarray

def plot_any(
    data: Union[Array, Tuple[Array, ...]],
    *,
    t: Optional[Array] = None,
    x: Optional[Array] = None,
    y: Optional[Array] = None,
    mode: str = "auto",          # "auto", "static", "animate"
    trail: int = 0,               # lunghezza scia per traiettorie (frame); 0 = nessuna
    labels: Dict[str,str] = None, # {"title": "...", "x": "...", "y": "...", "legend": "..."}
    figsize: Tuple[int,int] = (7,4),
    fps: int = 30,
    interval: Optional[int] = None,  # ms; se None calcolato da fps
    clim: Optional[Tuple[float,float]] = None, # limiti colore per campi 2D
    axis_equal: bool = False,
    xlim: Optional[Tuple[float,float]] = None,
    ylim: Optional[Tuple[float,float]] = None,
    save: Optional[str] = None,   # ".mp4" o ".gif"; se None -> solo show
    show: bool = True,
):
    """
    Una funzione per TUTTO:
    - Series 1D statiche: data.shape == (N,)    -> linea
    - Più serie statiche: data.shape == (N,M)   -> M linee
    - Profilo 1D nel tempo: data.shape == (T,N) (asse 0 = tempo) + x -> animazione linea(y(x,t))
    - Traiettorie 2D: data.shape == (T, P, 2)
    - Traiettorie 3D: data.shape == (T, P, 3)
    - Campo 2D nel tempo (PDE): data.shape == (T, Ny, Nx) (+ x,y opzionali) -> imshow animato
    - Fase/parametriche: passa una tupla (x(t), y(t)) o (x(t), y(t), z(t))
    - Campo vettoriale nel tempo: passa una tupla (U, V) con shape (T, Ny, Nx) ciascuno -> quiver animato

    Convenzione: l'ASSE 0 è sempre il tempo per animazioni (T = num. frame).
    """
    if labels is None: labels = {}
    interval = interval if interval is not None else int(1000 / fps)

    # Helper
    def _ensure_2d(a: Array) -> Array:
        return a if a.ndim == 2 else a.reshape(-1, 1)

    def _title_for_frame(i: int) -> str:
        if t is not None:
            return f"{labels.get('title','')}".strip() + (f"  (t={t[i]:.4g})" if labels.get('title') else f"t={t[i]:.4g}")
        return labels.get('title','')

    # --- Se l'utente passa una tupla per casi speciali ---
    if isinstance(data, tuple):
        # (x(t), y(t)) o (x(t), y(t), z(t)) parametriche
        if len(data) in (2,3) and all(isinstance(d, np.ndarray) and d.ndim == 1 for d in data):
            if len(data) == 2:
                X, Y = data
                fig, ax = plt.subplots(figsize=figsize)
                line, = ax.plot([], [], lw=2)
                ax.set_xlabel(labels.get("x","x"))
                ax.set_ylabel(labels.get("y","y"))
                if xlim: ax.set_xlim(*xlim)
                else:    ax.set_xlim(np.min(X), np.max(X))
                if ylim: ax.set_ylim(*ylim)
                else:    ax.set_ylim(np.min(Y), np.max(Y))
                if axis_equal: ax.set_aspect("equal", adjustable="box")

                def init(): 
                    line.set_data([], [])
                    ax.set_title(labels.get("title",""))
                    return line,
                def update(i):
                    line.set_data(X[:i+1], Y[:i+1])
                    ax.set_title(_title_for_frame(i))
                    return line,
                ani = FuncAnimation(fig, update, frames=len(X), init_func=init, interval=interval, blit=True)
            else:
                from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
                X, Y, Z = data
                fig = plt.figure(figsize=figsize)
                ax = fig.add_subplot(111, projection='3d')
                line, = ax.plot([], [], [], lw=2)
                ax.set_xlabel(labels.get("x","x"))
                ax.set_ylabel(labels.get("y","y"))
                ax.set_zlabel(labels.get("z","z"))
                ax.set_xlim(np.min(X), np.max(X))
                ax.set_ylim(np.min(Y), np.max(Y))
                ax.set_zlim(np.min(Z), np.max(Z))
                def init():
                    line.set_data([], [])
                    line.set_3d_properties([])
                    ax.set_title(labels.get("title",""))
                    return line,
                def update(i):
                    line.set_data(X[:i+1], Y[:i+1])
                    line.set_3d_properties(Z[:i+1])
                    ax.set_title(_title_for_frame(i))
                    return line,
                ani = FuncAnimation(fig, update, frames=len(X), init_func=init, interval=interval, blit=True)
        # Campo vettoriale (U,V) su griglia
        elif len(data) == 2 and all(isinstance(d, np.ndarray) and d.ndim == 3 for d in data):
            U, V = data  # shape (T, Ny, Nx)
            T, Ny, Nx = U.shape
            Xg, Yg = (np.meshgrid(x, y) if (x is not None and y is not None)
                      else np.meshgrid(np.arange(Nx), np.arange(Ny)))
            fig, ax = plt.subplots(figsize=figsize)
            q = ax.quiver(Xg, Yg, U[0], V[0], pivot="mid")
            ax.set_xlabel(labels.get("x","x"))
            ax.set_ylabel(labels.get("y","y"))
            if axis_equal: ax.set_aspect("equal", adjustable="box")
            if xlim: ax.set_xlim(*xlim)
            if ylim: ax.set_ylim(*ylim)
            def update(i):
                q.set_UVC(U[i], V[i])
                ax.set_title(_title_for_frame(i))
                return q,
            ani = FuncAnimation(fig, update, frames=T, interval=interval, blit=False)
        else:
            raise ValueError("Tupla non riconosciuta. Usa (x,y), (x,y,z) o (U,V) con asse 0=tempo.")
    else:
        a = np.asarray(data)

        # --- STATICHE ---
        if mode == "static" or a.ndim == 1 or (a.ndim == 2 and a.shape[0] < 5 and (t is None)):
            fig, ax = plt.subplots(figsize=figsize)
            if a.ndim == 1:
                xx = x if x is not None else np.arange(a.size)
                ax.plot(xx, a, lw=2, label=labels.get("legend"))
            else:
                # (N, M) -> M serie
                A = _ensure_2d(a)
                xx = x if x is not None else np.arange(A.shape[0])
                for j in range(A.shape[1]):
                    ax.plot(xx, A[:,j], lw=1.8, label=f"{labels.get('legend','y')}[{j}]")
                ax.legend(loc="best")
            ax.set_xlabel(labels.get("x","x"))
            ax.set_ylabel(labels.get("y","y"))
            ax.set_title(labels.get("title",""))
            if xlim: ax.set_xlim(*xlim)
            if ylim: ax.set_ylim(*ylim)
            ani = None

        # --- ANIMAZIONI ---
        else:
            # Profilo 1D nel tempo: (T, N)
            if a.ndim == 2:
                Tframes, N = a.shape
                xx = x if x is not None else np.arange(N)
                fig, ax = plt.subplots(figsize=figsize)
                line, = ax.plot(xx, a[0], lw=2)
                ax.set_xlabel(labels.get("x","x"))
                ax.set_ylabel(labels.get("y","y"))
                if xlim: ax.set_xlim(*xlim)
                else:    ax.set_xlim(np.min(xx), np.max(xx))
                if ylim: ax.set_ylim(*ylim)
                else:
                    vmin, vmax = np.min(a), np.max(a)
                    if vmin == vmax: vmin, vmax = vmin - 1, vmax + 1
                    ax.set_ylim(vmin, vmax)
                def update(i):
                    line.set_ydata(a[i])
                    ax.set_title(_title_for_frame(i))
                    return line,
                ani = FuncAnimation(fig, update, frames=Tframes, interval=interval, blit=True)

            # Campo 2D nel tempo: (T, Ny, Nx)
            elif a.ndim == 3 and a.shape[-1] not in (2,3):
                Tframes, Ny, Nx = a.shape
                Xg, Yg = (np.meshgrid(x, y) if (x is not None and y is not None)
                          else np.meshgrid(np.arange(Nx), np.arange(Ny)))
                fig, ax = plt.subplots(figsize=figsize)
                norm = Normalize(*clim) if clim is not None else None
                im = ax.pcolormesh(Xg, Yg, a[0], shading="auto", norm=norm)
                cb = plt.colorbar(im, ax=ax)
                cb.set_label(labels.get("cbar","u"))
                ax.set_xlabel(labels.get("x","x"))
                ax.set_ylabel(labels.get("y","y"))
                if axis_equal: ax.set_aspect("equal", adjustable="box")
                if xlim: ax.set_xlim(*xlim)
                if ylim: ax.set_ylim(*ylim)
                def update(i):
                    im.set_array(a[i].ravel())
                    ax.set_title(_title_for_frame(i))
                    return im,
                ani = FuncAnimation(fig, update, frames=Tframes, interval=interval, blit=False)

            # Traiettorie 2D/3D: (T, P, 2/3)
            elif a.ndim == 3 and a.shape[-1] in (2,3):
                Tframes, P, D = a.shape
                if D == 2:
                    fig, ax = plt.subplots(figsize=figsize)
                    sc = ax.scatter(a[0,:,0], a[0,:,1], s=25)
                    if axis_equal: ax.set_aspect("equal", adjustable="box")
                    ax.set_xlabel(labels.get("x","x"))
                    ax.set_ylabel(labels.get("y","y"))
                    if xlim: ax.set_xlim(*xlim)
                    else: ax.set_xlim(np.min(a[:,:,0]), np.max(a[:,:,0]))
                    if ylim: ax.set_ylim(*ylim)
                    else: ax.set_ylim(np.min(a[:,:,1]), np.max(a[:,:,1]))
                    # opzionale scia
                    if trail > 0:
                        lines = [ax.plot([], [], lw=1, alpha=0.6)[0] for _ in range(P)]
                    else:
                        lines = []
                    def update(i):
                        sc.set_offsets(a[i,:,:2])
                        if lines:
                            i0 = max(0, i-trail)
                            for p in range(P):
                                lines[p].set_data(a[i0:i+1,p,0], a[i0:i+1,p,1])
                        ax.set_title(_title_for_frame(i))
                        return (sc, *lines)
                    ani = FuncAnimation(fig, update, frames=Tframes, interval=interval, blit=False)
                else:
                    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
                    fig = plt.figure(figsize=figsize)
                    ax = fig.add_subplot(111, projection='3d')
                    sc = ax.scatter(a[0,:,0], a[0,:,1], a[0,:,2], s=20)
                    ax.set_xlabel(labels.get("x","x"))
                    ax.set_ylabel(labels.get("y","y"))
                    ax.set_zlabel(labels.get("z","z"))
                    ax.set_xlim(np.min(a[:,:,0]), np.max(a[:,:,0]))
                    ax.set_ylim(np.min(a[:,:,1]), np.max(a[:,:,1]))
                    ax.set_zlim(np.min(a[:,:,2]), np.max(a[:,:,2]))
                    def update(i):
                        pts = a[i]
                        sc._offsets3d = (pts[:,0], pts[:,1], pts[:,2])
                        ax.set_title(_title_for_frame(i))
                        return sc,
                    ani = FuncAnimation(fig, update, frames=Tframes, interval=interval, blit=False)
            else:
                raise ValueError(f"Forma array non riconosciuta per animazione: {a.shape}")

    # --- Salvataggio/visualizzazione ---
    if save:
        ext = os.path.splitext(save)[1].lower()
        if ext == ".gif":
            try:
                ani.save(save, writer="pillow", fps=fps)
            except Exception as e:
                print("Errore salvataggio GIF:", e)
        elif ext == ".mp4":
            try:
                ani.save(save, writer="ffmpeg", fps=fps)
            except Exception as e:
                print("Errore salvataggio MP4 (ti serve ffmpeg):", e)
        else:
            print("Estensione non supportata per salvataggio. Usa .gif o .mp4")
    if show:
        plt.show()

    return ani if 'ani' in locals() else None
