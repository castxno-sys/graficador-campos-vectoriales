import numpy as np
import matplotlib.pyplot as plt

# Malla en el dominio [-2, 2] x [-2, 2]
x = np.linspace(-2, 2, 17)
X, Y = np.meshgrid(x, x)

# (b) F = grad f = <2xy, x^2 - 3y^2>
plt.figure()
plt.quiver(X, Y, 2*X*Y, X**2 - 3*Y**2)              # flechas de F
plt.contour(X, Y, X**2*Y - Y**3, [0], colors="k")   # curva f = 0
plt.quiver(1, 1, 2, -2, color="orange")             # F(1,1) = <2,-2>
plt.plot([0, 2], [0, 2], "b--")                     # recta y = x
plt.title("F = <2xy, x^2 - 3y^2>")

# (c) G = <-y, x>
plt.figure()
plt.quiver(X, Y, -Y, X)                             # flechas de G
plt.title("G = <-y, x>")

plt.show()
