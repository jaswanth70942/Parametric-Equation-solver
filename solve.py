import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.optimize import minimize, differential_evolution
from scipy.interpolate import UnivariateSpline


# 1. Firstly, loading the csv file from the folder and sorting using PCA( principal component analysis).

df = pd.read_csv("/content/xy_data.csv")
points_raw = df[['x','y']].values

pca = PCA(n_components=1)
proj = pca.fit_transform(points_raw).flatten()
idx = np.argsort(proj)

points = points_raw[idx]
x_data = points[:,0]
y_data = points[:,1]


# 2. Performing arc length reconstruction to find parameter t.

dist = np.sqrt(np.sum(np.diff(points,axis=0)**2,axis=1))
s = np.hstack([[0], np.cumsum(dist)])
t_data = 6 + 54*(s / s[-1])


# 3. The given equations in the question

def model(params, t):
    theta, M, X = params
    e = np.exp(M*np.abs(t))
    osc = np.sin(0.3*t)
    x = t*np.cos(theta) - e*osc*np.sin(theta) + X
    y = 42 + t*np.sin(theta) + e*osc*np.cos(theta)
    return x, y

# -----------------------------------------------------------
# 4. Losses Calculation
# -----------------------------------------------------------
def L1(params):
    xp, yp = model(params, t_data)
    return np.mean(np.abs(xp-x_data) + np.abs(yp-y_data))

def L2(params):
    xp, yp = model(params, t_data)
    return np.sum((xp-x_data)**2 + (yp-y_data)**2)


# 5. Updating Bounds.

bounds = [
    (np.deg2rad(0), np.deg2rad(50)),
    (-0.05, 0.05),
    (0, 100)
]


# 6. Firstly Calculating Global L2

res_de = differential_evolution(L2, bounds, maxiter=600, polish=False)
params = res_de.x


# 7. Secondly, Calculating Local L2

res_l2 = minimize(L2, params, bounds=bounds, method='L-BFGS-B')
params = res_l2.x

# 8. Now calculating L1 using powell optimization method (better for L1)

res_l1 = minimize(L1, params, bounds=bounds, method='Powell')
params = res_l1.x

# 9. Finally Iterative T-Refinement, Splines, Smoothing.

for it in range(6):
    new_t = []
    for i in range(len(x_data)):
        def f(t):
            xp, yp = model(params, np.array([t]))
            return abs(xp - x_data[i]) + abs(yp - y_data[i])
        r = minimize(f, t_data[i], bounds=[(6,60)], method='Powell')
        new_t.append(r.x[0])

    new_t = np.array(new_t)

    # enforce monotonicity
    new_t = np.maximum.accumulate(new_t)

    # spline smooth the t-values → crucial for lowering L1 further
    spline = UnivariateSpline(np.arange(len(new_t)), new_t, s=len(new_t)*0.02)
    t_data = spline(np.arange(len(new_t)))

    # refit using L1 (Powell)
    params = minimize(L1, params, method='Powell', bounds=bounds).x
    print(f"Iteration {it+1}: L1 = {L1(params)}")


# 10. Results.

theta_final, M_final, X_final = params
print("\nFINAL RESULTS")
print("θ =", np.rad2deg(theta_final))
print("M =", M_final)
print("X =", X_final)
print("Final L1 =", L1(params))


# 11. Plotting and visualization

xp, yp = model(params, t_data)

plt.figure(figsize=(10,7))

# Actual data (scatter)
plt.scatter(x_data, y_data, s=10, color='blue', alpha=0.7, label='Actual Data')

# Fitted curve (smooth line)
plt.plot(xp, yp, color='red', linewidth=2.5, label='Fitted Curve')

plt.title("Actual Data vs Fitted Parametric Curve", fontsize=16)
plt.xlabel("X", fontsize=14)
plt.ylabel("Y", fontsize=14)

plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.4)
plt.axis('equal')

plt.tight_layout()
plt.show()
