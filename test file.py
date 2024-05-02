# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 20:32:56 2024

@author: YXAVION
"""

import matplotlib.pyplot as plt
import numpy as np
import math
import scipy
import random

rad = 1000
num_pts = 100
num_rot = 5


#%%


# default will be 5 rotations, ie 10pi
# default a = 0

# parametric equation
#x(θ) = (a + bθ) cos θ,
#y(θ) = (a + bθ) sin θ



#%%

a = rad/math.sqrt(2)
t = np.linspace(0, 2*np.pi, num = num_pts)

x = a * np.sqrt(2) * np.cos(t) / (np.sin(t)**2 + 1)
y = a * np.sqrt(2) * np.cos(t) * np.sin(t) / (np.sin(t)**2 + 1)

wg_pos = np.array([x, y, np.zeros(np.size(t)])

plt.scatter(wg_pos[:, 0], wg_pos[:, 1])













