"""Plot only the two saved trajectories, without re-reading source ephemerides."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from load_trajectories import load_trajectories

ROOT = Path(__file__).resolve().parent


def main():
    reference = load_trajectories()
    prediction = load_trajectories('sgp4')
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    fig = plt.figure(figsize=(13, 5.6), layout='constrained')
    orbit = fig.add_subplot(121, projection='3d')
    longitude = np.linspace(0, 2*np.pi, 48)
    latitude = np.linspace(0, np.pi, 24)
    earth_r = 6371
    orbit.plot_wireframe(earth_r*np.outer(np.cos(longitude), np.sin(latitude)),
                         earth_r*np.outer(np.sin(longitude), np.sin(latitude)),
                         earth_r*np.outer(np.ones_like(longitude), np.cos(latitude)),
                         color='#b8c8d3', alpha=.35, linewidth=.45)
    for name, color in [('Swarm-A', '#2074b4'), ('Swarm-C', '#d75b31')]:
        data = reference[name]
        r = data['position_m']/1000
        orbit.plot(*r.T, label=name, color=color, linewidth=1.7)
        i = np.flatnonzero(data['elapsed_seconds'] == 0)[0]
        orbit.scatter(*r[i], color=color, s=30)
    orbit.set(xlabel='GCRS x (km)', ylabel='GCRS y (km)', zlabel='GCRS z (km)',
              title='Two hours of shifted reference trajectories')
    orbit.set_box_aspect((1, 1, 1))
    orbit.view_init(elev=22, azim=-38)
    orbit.legend(loc='upper left')
    separation = fig.add_subplot(122)
    for data, label, color in [(reference, 'Reference trajectories', '#2074b4'),
                               (prediction, 'Uncorrected SGP4', '#d75b31')]:
        a, c = data['Swarm-A'], data['Swarm-C']
        t = a['elapsed_seconds']
        distance = np.linalg.norm(a['position_m']-c['position_m'], axis=1)
        mask = np.abs(t) <= 120
        separation.semilogy(t[mask], distance[mask], label=label, color=color, linewidth=1.8)
        i = np.flatnonzero(t == 0)[0]
        separation.scatter(0, distance[i], color=color, s=35, zorder=4)
    separation.axvline(0, color='#888888', linestyle=':', linewidth=1)
    separation.annotate('Reference: about 0.29 m', (0, .286), xytext=(20, .65),
                         arrowprops={'arrowstyle': '-', 'color': '#555555'}, fontsize=10)
    separation.set(xlabel='Seconds from reference crossing', ylabel='Separation (m; logarithmic scale)',
                   title='Separation near the constructed encounter')
    separation.grid(alpha=.18, which='both')
    separation.legend(loc='upper right')
    fig.suptitle('Swarm A + time-shifted Swarm C', fontsize=16, fontweight='bold')
    fig.savefig(ROOT/'trajectory_preview.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
