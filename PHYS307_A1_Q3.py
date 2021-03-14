# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 20:12:15 2021

@author: Ryan
"""

import numpy as np
import scipy.special as sc
import scipy.constants as const
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# Get first five values of x for which J_0(x)=0
cn = sc.jn_zeros(0,5)
c5 = cn[-1]
cn = cn[:-1]
c5sq = c5 ** 2

# Voltages associated with first four peaks, as read off of graph
V = np.array([-0.43, -0.375, -0.275, -0.15])

# Convert voltages to energies
eV = const.elementary_charge * V

# Electron mass in media
m = 0.38 * const.electron_mass

# Calculate R using the relation (1.7) from question text
R = np.sqrt(const.hbar ** 2 * (cn ** 2 - c5sq) / (2 * m * eV) )
Rmean = np.mean(R)

# Start off unnormalised.
A = 1

# Wave function with l=0, n=5. Normalisation controlled by A.
k = c5 / Rmean
def wave(r):
    return A*sc.jv(0, k*r)

def prob(r):
    return np.abs(wave(r)) ** 2

# Get normalisation factor
A = np.sqrt(1 / integrate.quad(lambda r: prob(r), -Rmean, Rmean)[0])

# Start plotting it all
# Following code based of example by Armin Moser
# https://matplotlib.org/stable/gallery/mplot3d/surface3d_radial.html
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# ax.axis('off')

# Create the mesh in polar coordinates and compute corresponding Z.
# Using this to avoid large colour bands in the central peak
r = np.concatenate((np.linspace(0, 0.1*Rmean, 40000),  np.linspace(0.1*Rmean, Rmean, 30000)))
# r = np.linspace(0, Rmean, 100000)
p = np.linspace(0, 2*np.pi, 50)
Ra, P = np.meshgrid(r, p, sparse=True)
Z = prob(Ra)

# Express the mesh in the cartesian system.
X, Y = Ra*np.cos(P), Ra*np.sin(P)

# Normalise data for colour mapping
norm = Normalize(Z.min(), Z.max())

# Plot the surface.
ax.plot_surface(X, Y, Z, cmap='viridis', norm=norm)
plt.show()