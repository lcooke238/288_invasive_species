'''
Created on Oct 22, 2017
Updated on March 19, 2018

@author: Sara
'''
from gurobipy import *
import numpy as np
import time
from PatrolProblem import PatrolProblem
import SinglePlannerNoPath as nopath
import argparse

#file1='../QENP_AnimalNonCom/PE_currentPatrol_pastPatrol_/Y_test_prob_predict_currentPatrol_pastPatrol_2017_2017_threshold7.5_6years_blackBoxFunction_detect_2.csv'
#file1='/Users/Sara/Documents/Euler/PathPlanning/MFNP_AnimalNonCom/Y_test_prob_predict_currentPatrol_pastPatrol_2017_2017_threshold7.5_6years_blackBoxFunction_detect.csv'
#'/Users/Sara/Documents/Euler/PathPlanning/QENP_AnimalNonCom/PE_currentPatrol_pastPatrol_/Y_test_prob_predict_currentPatrol_pastPatrol_2017_2017_threshold7.5_6years_blackBoxFunction_attack.csv'
#file2='../QENP_AnimalNonCom/PatrolPosts.csv'
#file2='./MFNP_AnimalNonCom/PatrolPosts.csv'

"""
r = 10
c = 10
n = r*c
r_s = r //2
c_s = c//2
T = 10
days = 15

start = time.time()
m = Model()

count_patrols=1
"""
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
        self.effort_increments = 10

    #Generate all the variables and constraints for the optimization
    def genSOSVars(self, m, x_vals, x):
        wx =[[0 for i in range(len(x_vals[j]))] for j in range(self.n)]
        for k in range(self.n):
            for i in range(len(x_vals[k])):
                wx[k][i] = m.addVar(lb=0, ub = 1.0, vtype=GRB.CONTINUOUS, name="wx%d-%d" %(k,i))
        m.update()


        for i in range(self.n):
            m.addConstr(x[i] == quicksum( wx[i][j] * x_vals[i][j] for j in range(len(x_vals[i]))))
            m.addConstr(1 == quicksum( wx[i][j] for j in range(len(x_vals[i]))))

        for k in range(self.n):
            if len(x_vals)>1:
                m.addSOS(GRB.SOS_TYPE2, wx[k])#,x_vals)
        m.update()

        return wx

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


    #Generates Flow variables and constraints for defined by graph
    def genPatrolVarsfromGraph(self, m, graph):
        x = {}
        f = [[[0 for t in range(self.T)] for j in range(self.n)] for i in range(self.n)]
        for i in range(self.n):
            x[i] = m.addVar(lb=0, ub = 10000, vtype=GRB.CONTINUOUS, name="x%d-%d" % (self.count_patrols,i))
        for i in range(self.n):
            for j in range(self.n):
                for t in range(self.T):
                        f[i][j][t] = m.addVar(lb=0, ub = 1.0, vtype=GRB.CONTINUOUS, name="f%d-%d-%d-%d" % (self.count_patrols,i,j,t))

        m.update()
        flat_list = []
        for sf in f:
            for ssf in sf:
                for item in ssf:
                    flat_list.append(item)

        m.addConstr(self.T >= quicksum(flat_list))

        # Add constraints
        m.addConstr(self.T*self.days >= quicksum( x[j] for j in range(self.n)))
        for k in range(self.n):
            fl = []
            fl.append(f[k][k])
            for i in graph.neighbors(k):
                fl.append(f[i][k])


                flat_fl = [val for sublist in fl for val in sublist]
            m.addConstr(x[k] == self.days*quicksum( flat_fl[p] for p in range(len(flat_fl))))

            for t in range(1,self.T):
                flin = []
                flout=[]
                for i in graph.neighbors(k):
                    flin.append(f[i][k][t-1])
                    flin.append(f[k][k][t-1])

                    flout.append(f[k][i][t])
                flin.append(f[k][k][t-1])
                flout.append(f[k][k][t])

                m.addConstr(quicksum( flout[p] for p in range(len(flout))) == quicksum( flin[p] for p in range(len(flin))))

            d=0
            if k==graph.source:
                d=1
            flin = []
            flout=[]
            flin.append(f[k][k][0])
            flout.append(f[k][k][self.T-1])

            for i in graph.neighbors(k):
                flin.append(f[k][i][0])
                flout.append(f[i][k][self.T-1])

            m.addConstr(d == quicksum( flin[j] for j in range(len(flin))))
            m.addConstr(quicksum( flout[j] for j in range(len(flout))) == d)

        fl = []
        k = graph.source
        for i in graph.neighbors(k):
            fl.append(f[k][i][0])
        #m.addConstr(quicksum( fl[j] for j in range(len(fl))) == 1)


        flin=[]
        flout=[]
        for i in graph.neighbors(k):
            flout.append(f[i][k][self.T-1])
        #m.addConstr(quicksum( flout[j] for j in range(len(flout))) == 1)


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

    def solvePost(self, i, g, data, effortx, var=None, robust=False):
        if var != None:
             g.graph.datafolder += "var"
             if robust:
                g.graph.datafolder += "robust"

        start = time.time()
        m = Model()
        #self.n=g.n


        # Add variables
        #x,f = self.genPatrolVarsfromGraph(m, g.graph)
        x,f = self.genPatrolVars(m)
        #needed?
        m.update()

        obj = self.setObjectiveandSolve(m, data, effortx, x, var=var, robust=robust)
        end = time.time()
        totaltime = end - start

        solx1 , solf1, obj = self.getSol(m, x, f)
        soldata = [i, totaltime, obj]
        soldata.extend(solx1)
        #g.graph.plotUniformPredictions(data, i, 0, effortx)
        #g.graph.plotUniformPredictions(data, i, 1, effortx)
        #g.graph.plotUniformPredictions(data, i, 2, effortx)
        #g.graph.plotUniformPredictions(data, i, 10,effortx)

        g.graph.writeSolution("qeen_4.csv", soldata)
        g.graph.plotHeatGraph(solx1, i)
        g.graph.plotObjHeatGraph(solx1, effortx, data, i, "", objective=obj)

        g.graph.samplePatrols(solf1, self.days, post=i)


        return obj, max(solx1)

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
        solx = [solx1[i]/300.0 for i in range(len(solx1))]
        return solx
        #look at these outputs
        
    def runExperiments(self, g, file1, file2, file3=None, datafolder='./kai_data/test/GP_'):
        print("running experiments...")
        for i in range(1,2):
            m = Model()
            data, effortx = g.loadFile1D(file1, file2, i, points=10, datafolder=datafolder)
            if file3 != None:
                var, effortx = g.loadFile1D(file3, file2, i, points=10, datafolder=datafolder)
            self.n = g.n


            # Add variables
            x,f = self.genPatrolVarsfromGraph(m, g.graph)
            m.update()
            start = time.time()
            print(len(effortx[0]))

            obj = self.setObjectiveandSolve(m, data, effortx, x)
            end = time.time()
            totaltime = end - start
            solx1 , solf1, obj = self.getSol(m, x, f)
            soldata = [i, totaltime, obj]
            soldata.extend(solx1)
            # g.graph.plotUniformPredictions(data, i, 0, effortx)
            # g.graph.plotUniformPredictions(data, i, 1, effortx)
            # g.graph.plotUniformPredictions(data, i, 2, effortx)
            # g.graph.plotUniformPredictions(data, i, 3, effortx)
            # g.graph.plotUniformPredictions(data, i, 4, effortx)
            # g.graph.plotUniformPredictions(data, i, 5, effortx)
            # g.graph.plotUniformPredictions(data, i, 6, effortx)
            # g.graph.plotUniformPredictions(data, i, 7, effortx)
            # g.graph.plotUniformPredictions(data, i, 8, effortx)

            # g.graph.plotUniformRiskPredictions(data, i, effortx)

            g.graph.writeSolution("qeen_4.csv", soldata)
            g.graph.plotHeatGraph(solx1, i)
            g.graph.plotObjHeatGraph(solx1, effortx, data, i, "", objective=obj)
            g.graph.samplePatrols(solf1, self.days, post=i)

            plannernp = nopath.SinglePlannerNoPath(days=days, T=T)
            print(max(solx1))
            plannernp.solvePostFromSol( g, data, effortx, i, vmax=max(solx1))
            m.terminate()

    def runRiskMap(self, file1, file2, file3=None, datafolder='./kai_data/test/GP_', resolution=1000):
        print("running risk maps...")
        r = 400
        c = 400
        g = PatrolProblem(T, r, c, r_s, c_s, obj="max", resolution=resolution, method=self.method)
        data, effortx = g.loadFile1D(file1, file2, i, points=10, datafolder=datafolder)
        if file3 != None:
            var, effortx = g.loadFile1D(file3, file2, i, points=10, datafolder=datafolder)
        self.n=g.n

        g.graph.plotUniformPredictions(data, i, 0, effortx)
        g.graph.plotUniformPredictions(data, i, 1, effortx)
        g.graph.plotUniformPredictions(data, i, 2, effortx)
        g.graph.plotUniformPredictions(data, i, 3, effortx)
        g.graph.plotUniformPredictions(data, i, 4, effortx)
        g.graph.plotUniformPredictions(data, i, 5, effortx)
        g.graph.plotUniformPredictions(data, i, 6, effortx)
        g.graph.plotUniformPredictions(data, i, 7, effortx)
        g.graph.plotUniformPredictions(data, i, 8, effortx)

        g.graph.plotUniformRiskPredictions(data, i, effortx)

if __name__ == "__main__":
    planner = SinglePlanner()
    # planner.test()
    
    """
    parser = argparse.ArgumentParser(description='Bagging Cross Validation Blackbox function')
    parser.add_argument('-r', '--resolution', default=1000, help='Input the resolution scale')
    parser.add_argument('-p', '--park', help='Input the park name', required=True)
    parser.add_argument('-x', '--xaxis', default=16, help='Input the x axis length')
    parser.add_argument('-y', '--yaxis', default=16, help='Input the y axis length')
    parser.add_argument('-T', '--Tlength', default=10, help='Input the max patrol effort')
    parser.add_argument('-d', '--days', default=2, help='Input the number of patrollers')
    parser.add_argument('-m', '--method', help='Input the method')
    parser.add_argument('-simple', '--simple', default=False, help='1 => without using ensemble, 0 => using ensemble')
    parser.add_argument('-cutoff', '--cutoff', default=0, help='Input the cutoff threshold of patrol effort')

    args = parser.parse_args()

    park = args.park
    method = args.method
    simple_classification = args.simple
    file_folder = '../../{0}_datasets'.format(park)
    resolution = int(args.resolution)
    cutoff_threshold = float(args.cutoff)

    #file1='../QENP_AnimalNonCom/PE_currentPatrol_pastPatrol_/Y_test_prob_predict_currentPatrol_pastPatrol_2017_2017_threshold7.5_6years_blackBoxFunction_detect_2.csv'
    #file1 = '../Gonarezhu_datasets/test_2015/BG/Illegal_Activity/thrshOverNegLabels/3Month/TrainedModels/PE_currentPatrol_pastPatrol_/Y_test_prob_predict_currentPatrol_pastPatrol_2015_2015_threshold21.0_3years_blackBoxFunction_detect.csv'
    #file1 = file_folder + 'test_2018/BG/Illegal_Activity/thrshOverNegLabels/3Month/TrainedModels/PE_currentPatrol_pastPatrol_/Y_test_prob_predict_currentPatrol_pastPatrol_2018_2018_threshold15_2years_blackBoxFunction_detect.csv'
    if simple_classification:
        file1 = file_folder + '/resolution/{0}m/{1}/simple_output_{2}/Prob/Merged/Y_test_prob_blackBoxFunction_detect.csv'.format(str(resolution), method, cutoff_threshold)
    else:
        file1 = file_folder + '/resolution/{0}m/{1}/output_{2}/Prob/Merged/Y_test_prob_blackBoxFunction_detect.csv'.format(str(resolution), method, cutoff_threshold)
    # file1='./GP_detect_data.csv'
    #file2='../QENP_AnimalNonCom/PatrolPosts.csv'
    file2 = file_folder + '/PatrolPosts.csv'
    # file3='./queen_var_detect.csv'

    output_path = '{0}/resolution/{1}m/{2}/output/'.format(file_folder, str(resolution), method)
    exp_output_path = '{0}_exp/'.format(output_path[:-1])
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(exp_output_path):
        os.makedirs(exp_output_path)

    r = int(args.xaxis)
    c = int(args.yaxis)
    n = r*c
    r_s = r//2
    c_s = c//2
    T = int(args.Tlength)
    days = int(args.days)

    g = PatrolProblem(T, r, c, r_s, c_s, obj="max", resolution=resolution)
    g.max_effort = T*days
    g.days = days
    i = 1
    points = 10
    data, effortx = g.loadFile1D(file1, file2, i, points=points,ceiling=False, datafolder=output_path)
    # var, effortx = g.loadFile1D(file3, file2, i, points=points,ceiling=False)

    g.graph.plotGraph()
    planner = SinglePlanner(days=days, T=T, method=method)
    obj1, max_solx1 = planner.solvePost(i, g, data, effortx)
    # planner.solvePost(16, g, data, effortx, var=var)
    # planner.solvePost(16, g, data, effortx, var=var, robust=True)


    plannernp = nopath.SinglePlannerNoPath(days=days, T=T)
    #print max_solx1
    #plannernp.solvePostFromSol( g, data, effortx, i, vmax=max_solx1)
    obj2 = plannernp.solvePostFromSol( g, data, effortx, i, vmax=max_solx1)



    #planner.solvePost(17)
    planner.runExperiments(g, file1, file2, datafolder=exp_output_path)
    planner.runRiskMap(file1, file2, datafolder=exp_output_path, resolution=resolution)
    """
    
"""
g = PatrolProblem(T, r, c, r_s, c_s,obj="max")
g.max_effort=T*days
g.days=days

#data, effortx = g.loadFile1(file1, file2, 3)
data, effortx = g.loadFileGrid(file1, file2, 3)
#g.plotData(r_s*c+c_s)
n=g.n


# Add variables
x,f = genPatrolVars(m)
#x,f = genPatrolVarsfromGraph(m, g.graph)
m.update()


#wx = genSOSVars(effortx, x)
setObjectiveandSolve(m, effortx, x)

end = time.time()
print "Time: ", end-start

solx1 , solf1 = getSol(m, x, f)
g.writeSolution("test.csv", solx1)
g.graph.plotHeatGraph(solx1)
g.plotHeatMap(solx1)
g.plotPatrol(solf1)
"""
