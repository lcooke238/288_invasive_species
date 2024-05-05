# adapted fairly directly from the PAWS framework used in https://github.com/lily-x/PAWS-public/tree/master/planning

from gurobipy import *
import numpy as np

class SinglePlanner(object):
    def __init__(self, graph=None, M=25, N=25, days=15, T=20, method=None):
        self.r = M
        self.c = N
        self.n = self.r * self.c
        self.r_s = self.r//2
        self.c_s = self.c//2
        self.T = T
        self.days = days
        self.count_patrols = 1
        self.method = method

    #Generates Flow Variables and Constraints for a nxn grid graph
    def genPatrolVars(self, m):
        x = {}
        f = [[[0 for t in range(self.T)] for j in range(self.n)] for i in range(self.n)]
        for i in range(self.n):
            x[i] = m.addVar(lb=0, ub=10000, vtype=GRB.CONTINUOUS, name="x%d-%d" % (self.count_patrols, i))
        for i in range(self.n):
            for j in range(self.n):
                for t in range(self.T):
                        f[i][j][t] = m.addVar(lb=0, ub=1.0, vtype=GRB.CONTINUOUS, name="f%d-%d-%d-%d" % (self.count_patrols, i, j, t))

        m.update()
        flat_list = []
        for sf in f:
            for ssf in sf:
                for item in ssf:
                    flat_list.append(item)

        m.addConstr(self.T >= quicksum(flat_list))

        # Add constraints
        m.addConstr(self.T*self.days >= quicksum( x[j] for j in range(self.n)))
        for i in range(self.r):
            for j in range(self.c):
                k = i*self.c+j
                fl = []
                if j<(self.c-1): fl.append(f[i*self.c+j+1][k])
                if j>0: fl.append(f[i*self.c+j-1][k])
                if i<(self.r-1): fl.append(f[(i+1)*self.c+j][k])
                if i>0: fl.append(f[(i-1)*self.c+j][k])
                flat_fl = [val for sublist in fl for val in sublist]
                m.addConstr(x[k] == self.days*quicksum( flat_fl[p] for p in range(len(flat_fl))))

                for t in range(1,self.T):
                    flin = []
                    if j<(self.c-1): flin.append(f[i*self.c+j+1][k][t-1])
                    if j>0: flin.append(f[i*self.c+j-1][k][t-1])
                    if i<(self.r-1): flin.append(f[(i+1)*self.c+j][k][t-1])
                    if i>0: flin.append(f[(i-1)*self.c+j][k][t-1])
                    flin.append(f[k][k][t-1])

                    flout=[]
                    if j<self.c-1: flout.append(f[k][i*self.c+j+1][t])
                    if j>0: flout.append(f[k][i*self.c+j-1][t])
                    if i<self.r-1: flout.append(f[k][(i+1)*self.c+j][t])
                    if i>0: flout.append(f[k][(i-1)*self.c+j][t])
                    flout.append(f[k][k][t])

                    m.addConstr(quicksum( flout[p] for p in range(len(flout))) == quicksum( flin[p] for p in range(len(flin))))
        for i in range(self.r):
            for j in range(self.c):
                d=0
                k = i*self.c+j
                if i==self.r_s and j==self.c_s:
                    d=1
                flin = []
                if j<(self.c-1): flin.append(f[k][i*self.c+j+1][0])
                if j>0: flin.append(f[k][i*self.c+j-1][0])
                if i<(self.r-1): flin.append(f[k][(i+1)*self.c+j][0])
                if i>0: flin.append(f[k][(i-1)*self.c+j][0])
                flin.append(f[k][k][0])
                m.addConstr(d == quicksum( flin[j] for j in range(len(flin))))

                flout=[]
                if j<self.c-1: flout.append(f[i*self.c+j+1][k][self.T-1])
                if j>0: flout.append(f[i*self.c+j-1][k][self.T-1])
                if i<self.r-1: flout.append(f[(i+1)*self.c+j][k][self.T-1])
                if i>0: flout.append(f[(i-1)*self.c+j][k][self.T-1])
                flout.append(f[k][k][self.T-1])
                m.addConstr(quicksum( flout[j] for j in range(len(flout))) == d)
        fl = []
        k = self.r_s*self.c+self.c_s
        print(k)
        if self.c_s<self.c-1: fl.append(f[k][self.r_s*self.c+self.c_s+1][0])
        if self.c_s>0: fl.append(f[k][self.r_s*self.c+self.c_s-1][0])
        if self.r_s<self.r-1: fl.append(f[k][(self.r_s+1)*self.c+self.c_s][0])
        if self.r_s>0: fl.append(f[k][(self.r_s-1)*self.c+self.c_s][0])
        m.addConstr(quicksum( fl[j] for j in range(len(fl))) == 1)


        flin=[]
        flout=[]
        if self.c_s<self.c-1: flout.append(f[self.r_s*self.c+self.c_s+1][k][self.T-1])
        if self.c_s>0: flout.append(f[self.r_s*self.c+self.c_s-1][k][self.T-1])
        if self.r_s<self.r-1: flout.append(f[(self.r_s+1)*self.c+self.c_s][k][self.T-1])
        if self.r_s>0: flout.append(f[(self.r_s-1)*self.c+self.c_s][k][self.T-1])
        m.addConstr(quicksum( flout[j] for j in range(len(flout))) == 1)


        m.update()
        return x, f


    def getSol(self, m, x, f):
        sol=[0 for i in range(self.n)]
        solf=[[[0 for t in range(self.T)] for j in range(self.n)] for i in range(self.n)]
        if m.status == GRB.Status.OPTIMAL:
            obj = m.getAttr('ObjVal')
            solx = m.getAttr('x', x)
            for i in range(self.n):
                for j in range(self.n):
                    for t in range(self.T):
                        solf[i][j][t] =  f[i][j][t].getAttr('x')
        for i in range(self.n):
            sol[i]=solx[i]
        print(np.max(solf))
        return sol, solf, obj

    def setObjectiveandSolve(self, m, data, effortx, x, var=None, robust=False):
        obj = []
        m.setObjective(0.0, GRB.MAXIMIZE)

        c = 0.05
        for i in range(self.n):
            if len(effortx[i]) > 1:
                m.setPWLObj(x[i], effortx[i], data[i][:len(effortx[i])])

                if var != None:
                    if robust:
                        y = [ -c*v for v in var[i][:len(effortx[i])]]
                        m.setPWLObj(x[i], effortx[i], y)
                    else:
                        y = [ c*v for v in var[i][:len(effortx[i])]]
                        m.setPWLObj(x[i], effortx[i], y)


        m.update()
        #m.write("out.lp")
        m.optimize()
        #m.write("out.sol")

    # effortx and data are x and y values for the piecewise linearization of the objective
    # see Model.setPWLObj in the gorubi documentation
    # in our case, effortx will be list of x-values corresponding to discretization of effort
    # y value will be the predicted probability of catching the invasive species
    def test(self, effortx, data):
        m = Model()
        x,f = self.genPatrolVars(m)
        m.update()
        obj = self.setObjectiveandSolve(m, data, effortx, x)
        solx1, solf1, obj = self.getSol(m,x,f)
        return solx1
        #look at these outputs
        
