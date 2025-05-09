import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

def f(x,y):
    return 2*(x*(1-x) + y*(1-y))
def v(x,y):
    return x*(1-x)*y*(1-y)
def g(x,y):
    return x**3-3*x*y**2
def u(x,y):
    return v(x,y) + g(x,y)

#Gebruikmakend van de poissongenerator in githubpagina
data = loadmat('xF.mat')
xF = data['xF']
xF = xF.tolist()
xF = np.array(xF)

data = loadmat('yF.mat')
yF = data['yF']
yF = yF.tolist()
yF = np.array(yF)

data = loadmat('xg.mat')
xG = data['xG']
xG = xG.tolist()
xG = np.array(xG)

data = loadmat('yG.mat')
yG = data['yG']
yG = yG.tolist()
yG = np.array(yG)

data = loadmat('sensor_int.mat')
sensor_int = data['sensor_int']
sensor_int = sensor_int.tolist()
sensor_int = np.array(sensor_int)

data = loadmat('sensor_ext.mat')
sensor_ext = data['sensor_ext']
sensor_ext = sensor_ext.tolist()
sensor_ext = np.array(sensor_ext)

m = len(sensor_int)**(0.5)


XF = (xF,sensor_int)
XG = (xG,sensor_int)
branch_f = [f(x,y) for x, y in sensor_int]
data_f = dde.data.TripleCartesianProd(
    X_train=XF, y_train=yF, X_test=XF, y_test=yF
)
p_f = int(m**2)
net_f = dde.nn.DeepONetCartesianProd(
    [p_f, p_f, p_f],
    [2, p_f, p_f],
    "sigmoid",
    "Glorot uniform",
)
model_f = dde.Model(data_f, net_f)
model_f.compile("adam", lr=0.01, metrics=["mean l2 relative error"])
losshistory, train_state = model_f.train(iterations=30000)

branch_g = [g(x,y) for x, y in sensor_ext]

data_g = dde.data.TripleCartesianProd(
    X_train=XG, y_train=yG, X_test=XG, y_test=yG
)
p_g = 4*int(m)+4
net_g = dde.nn.DeepONetCartesianProd(
    [p_g, p_g, p_g],
    [2, p_g, p_g],
    "sigmoid",
    "Glorot uniform",
)
model_g = dde.Model(data_g, net_g)
model_g.compile("adam", lr=0.01, metrics=["mean l2 relative error"])
losshistory, train_state = model_g.train(iterations=30000)

xy = [[x, y] for x in np.linspace(0,1,101) for y in np.linspace(0,1,101)]
U = [u(x, y) for x, y in xy]

xy = np.array(xy)
xi = xy[:, 0]
yi = xy[:, 1]

V = model_f.predict(([branch_f],xy))
W = model_g.predict(([branch_g],xy))

U_approx = V + W
U_approx = U_approx[0]
U_approx = U_approx.reshape((101, 101))
print("U_approx", U_approx)

V = V[0]
V = V.reshape((101, 101))
print("V", V)

W = W[0]
W = W.reshape((101, 101))
print("W", W)


Xi = xi.reshape((101, 101))
Yi = yi.reshape((101, 101))

U = np.array(U)
U = U.reshape((101, 101))

#Plots
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(Xi, Yi, U, cmap='viridis')
plt.xlabel("x")
plt.ylabel("y")
#plt.savefig('DONPDE1.png')
plt.show()

fig = plt.figure()
levels = 15
contourf = plt.contourf(Xi, Yi, U, levels=levels, cmap='viridis')
plt.xlabel('x')
plt.ylabel('y')
contour_lines = plt.contour(Xi, Yi, U, levels=levels, colors='black', linewidths=0.5)
plt.colorbar(contourf)
#plt.savefig('DONPDE1Contour.png')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(Xi, Yi, U_approx, cmap='viridis')
plt.xlabel("x")
plt.ylabel("y")
#plt.savefig('DONPDE2.png')
plt.show()

fig = plt.figure()
levels = 15
contourf = plt.contourf(Xi, Yi, U_approx, levels=levels, cmap='viridis')
plt.xlabel("x")
plt.ylabel("y")
contour_lines = plt.contour(Xi, Yi, U_approx, levels=levels, colors='black', linewidths=0.5)
plt.colorbar(contourf)
#plt.savefig('DONPDE2Contour.png')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(Xi, Yi, abs(U-U_approx), cmap='viridis')
plt.xlabel("x")
plt.ylabel("y")
#plt.savefig('DONPDE3.png')
plt.show()

fig = plt.figure()
levels = 15
contourf = plt.contourf(Xi, Yi, abs(U-U_approx), levels=levels, cmap='viridis')
plt.xlabel('x')
plt.ylabel('y')
contour_lines = plt.contour(Xi, Yi, abs(U-U_approx), levels=levels, colors='black', linewidths=0.5)
plt.colorbar(contourf)
#plt.savefig('DONPDE3Contour.png')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(Xi, Yi, V, cmap='viridis')
plt.xlabel("x")
plt.ylabel("y")
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(Xi, Yi, W, cmap='viridis')
plt.xlabel("x")
plt.ylabel("y")
plt.show()