import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
import math
def f1(x):
    return 3*x**2 + 2*x - 3
def F1(x):
    return x**3+x**2-3*x
def f2(x):
    return 15 *x**2 -20*x + 55/9
def F2(x):
    return 5*(x-1/3)*(x-2/3)*(x-1) + 10/9
def f3(x):
    return math.exp(x)
def F3(x):
    return math.exp(x)-1
def f4(x):
    return math.pi * math.cos(math.pi * x)
def F4(x):
    return math.sin(math.pi * x)
def f5(x):
    return 0
def F5(x):
    return 0
def f6(x):
    return -2
def F6(x):
    return -2*x
def trainingdata(samples,sensorAmount=51,degree_1=4,max_coeff=5):
    sensor = np.linspace(0,1,sensorAmount)                                          #sensoren instellen
    u_coeffs = np.random.uniform(-max_coeff, max_coeff, (samples, degree_1))        #willekeurige coëfficienten voor x^3 type functies
    u_samples = np.array([np.polyval(c, sensor) for c in u_coeffs])                 #We beschouwen de functiewaarde voor elke sensor
    U_coeffs = np.array([np.polyint(c) for c in u_coeffs])                          #coëfficienten voor de primitieve U de constante term is altijd 0 (geen +C) term
    U_samples = np.array([np.polyval(c, sensor) for c in U_coeffs])
    sensorReformat = np.array([[x] for x in sensor]) #1D
    return u_samples, sensorReformat, U_samples

def reformator(trunk_input,branch_input):
    #Dit verandert de structuur van het trunk- en branch input zodat DeepXDE ze kan herkennen
    if isinstance(trunk_input[0], (int,float)):
        trunkReformat = np.array([[x for x in trunk_input]])
    else:
        trunkReformat = np.array([x for x in trunk_input])
    if isinstance(branch_input, (int, float)):
        branch_input = [branch_input]
    branchReformat = np.array([[x] for x in branch_input])
    return trunkReformat, branchReformat

n = 51
Z = trainingdata(150,n,4,5)
X_train = Z[:2]
y_train = Z[2]

#We gebruiken X_train en y_train als testwaarden omdat we tegen niets willen vergelijken
#(deze argumenten weglaten geeft een error)
data = dde.data.TripleCartesianProd(
    X_train=X_train, y_train=y_train, X_test=X_train, y_test=y_train
)

#aantal neuronen op de laatste laag van het branch- en trunk neuraal netwerk
p = 51
net = dde.nn.DeepONetCartesianProd(
    [n, p, p],
    [1, p, p],
    "sigmoid", #/relu
    "Glorot uniform", #/He uniform
)
model = dde.Model(data, net)
model.compile("adam", lr=0.001, metrics=["mean l2 relative error"])
losshistory, train_state = model.train(iterations=100000)

#model.save("IntegrationModel")

f1_branch = list(map(f1,np.linspace(0,1,n)))
f2_branch = list(map(f2,np.linspace(0,1,n)))
f3_branch = list(map(f3,np.linspace(0,1,n)))
f4_branch = list(map(f4,np.linspace(0,1,n)))
f5_branch = list(map(f5,np.linspace(0,1,n)))
f6_branch = list(map(f6,np.linspace(0,1,n)))

x_grid = np.linspace(0,1,100)

output1 = model.predict(reformator(f1_branch,x_grid))
output2 = model.predict(reformator(f2_branch,x_grid))
output3 = model.predict(reformator(f3_branch,x_grid))
output4 = model.predict(reformator(f4_branch,x_grid))
output5 = model.predict(reformator(f5_branch,x_grid))
output6 = model.predict(reformator(f6_branch,x_grid))

plt.figure()
plt.plot(x_grid, list(map(F1,x_grid)))
plt.plot(x_grid, output1[0])
plt.xlabel("x")
plt.ylabel("y")
plt.legend(['F(x)','DeepONet'])
plt.grid()
#plt.savefig('DON1.png')
plt.show()

plt.figure()
plt.plot(x_grid, list(map(F2,x_grid)))
plt.plot(x_grid, output2[0])
plt.xlabel("x")
plt.ylabel("y")
plt.legend(['F(x)','DeepONet'])
plt.grid()
#plt.savefig('DON2.png')
plt.show()

plt.figure()
plt.plot(x_grid, list(map(F3,x_grid)))
plt.plot(x_grid, output3[0])
plt.xlabel("x")
plt.ylabel("y")
plt.legend(['F(x)','DeepONet'])
plt.grid()
#plt.savefig('DON3.png')
plt.show()

plt.figure()
plt.plot(x_grid, list(map(F4,x_grid)))
plt.plot(x_grid, output4[0])
plt.xlabel("x")
plt.ylabel("y")
plt.legend(['F(x)','DeepONet'])
plt.grid()
#plt.savefig('DON4.png')
plt.show()

plt.figure()
plt.plot(x_grid, list(map(F5,x_grid)))
plt.plot(x_grid, output5[0])
plt.xlabel("x")
plt.ylabel("y")
plt.legend(['F(x)','DeepONet'])
plt.grid()
#plt.savefig('DON5.png')
plt.show()

plt.figure()
plt.plot(x_grid, list(map(F6,x_grid)))
plt.plot(x_grid, output6[0])
plt.xlabel("x")
plt.ylabel("y")
plt.legend(['F(x)','DeepONet'])
plt.grid()
#plt.savefig('DON6.png')
plt.show()

