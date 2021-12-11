import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np

ngridx = 100
ngridy = 200

# define the hyperparameter space (epsilon = 1-20, C=1-10)
epsilon_min = 1
epsilon_max = 10
C_min = 1
C_max =10
epsilon = list(np.arange(epsilon_min,epsilon_max,1))
C = list(np.arange(C_min,C_max,1))

#calculate cv_score for each hyperparameter combination
cv_scores, c_settings, epsilon_settings = grid_search(epsilon, C)

#define plot dimensions
x_plot = c_settings
y_plot = epsilon_settings
z_plot = cv_scores

#define figure
fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(8, 8))
# -----------------------
# Interpolation on a grid
# -----------------------
# A contour plot of irregularly spaced data coordinates
# via interpolation on a grid.

# Create grid values first.
xi = np.linspace(min(x_plot)-1, max(x_plot)+1, ngridx)
yi = np.linspace(min(y_plot)-1, max(y_plot)+1, ngridy)

# Perform linear interpolation of the data (x,y)
# on a grid defined by (xi,yi)
triang = tri.Triangulation(x_plot, y_plot)
interpolator = tri.LinearTriInterpolator(triang, z_plot)
Xi, Yi = np.meshgrid(xi, yi)
zi = interpolator(Xi, Yi)

# Note that scipy.interpolate provides means to interpolate data on a grid
# as well. The following would be an alternative to the four lines above:
ax1.contour(xi, yi, zi, levels=14, linewidths=0.5, colors='k')
cntr1 = ax1.contourf(xi, yi, zi, levels=14, cmap="RdBu_r")

fig.colorbar(cntr1, ax=ax1)
ax1.plot(x, y, 'ko', ms=3)
ax1.set(xlim=(min(x), max(x)), ylim=(min(y), max(y)))
ax1.set_title('grid and contour (%d points, %d grid points)' %
              (npts, ngridx * ngridy))
ax1.set_xlabel('C')
ax1.set_ylabel('Epsilon')


# ----------
# Tricontour
# ----------
# Directly supply the unordered, irregularly spaced coordinates
# to tricontour.


# define the hyperparameter space (epsilon = 1-20, C=1-10)
epsilon = list(np.arange(1,10,1))
C = list(np.arange(5,10,1))
cv_scores, c_settings, epsilon_settings = grid_search(epsilon, C)

x_plot2 = c_settings
y_plot2 = epsilon_settings
z_plot2 = cv_scores

ax2.tricontour(x, y, z, levels=100, linewidths=0.5, colors='k')
cntr2 = ax2.tricontourf(x_plot2, y_plot2, z_plot2, levels=14, cmap="RdBu_r")

fig.colorbar(cntr2, ax=ax2)
ax2.plot(x_plot2, y_plot2, 'ko', ms=3)
ax2.set(xlim=(min(x_plot2), max(x_plot2)), ylim=(min(y_plot2), max(y_plot2)))
ax2.set_title('tricontour (%d points)' % npts)
ax2.set_xlabel('C')
ax2.set_ylabel('Epsilon')

plt.subplots_adjust(hspace=0.5)
plt.show()

from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np


def grid_search(epsilon, C):
    cv_scores = []
    c_settings = []
    epsilon_settings = []

    for n in range(len(epsilon)):
        for i in range(len(C)):
            ax = plt.subplot(1, 1, 1)
            plt.setp(ax, xticks=(), yticks=())

            # support vector regression
            pipeline = make_pipeline(StandardScaler(), SVR(C=C[i], epsilon=epsilon[n]))
            pipeline.fit(X[:, np.newaxis], y)

            # Evaluate the models using crossvalidation
            scores = cross_val_score(pipeline, X[:, np.newaxis], y, scoring="neg_mean_squared_error", cv=10)
            cv_scores.append(round(scores.mean(), 2))
            c_settings.append(C[i])
            epsilon_settings.append(epsilon[n])

    return cv_scores, c_settings, epsilon_settings

# plt.plot(X_test, true_fun(X_test), label="True function")
# plt.scatter(c_settings, cv_scores, edgecolor='b', s=20, label="Samples")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.xlim((0, 0.2))
# plt.ylim((-2, 2))
# plt.legend(loc="best")
# plt.title("Degree {}\nMSE = {:.2e}(+/- {:.2e})".format(
#    degrees[i], -scores.mean(), scores.std()))
# plt.show()

# plt.savefig("polynomial_regression_example.png", dpi=150)