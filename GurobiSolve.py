from gurobipy import Model, GRB, quicksum, Env


# gurobi function that solves the sample-based SSKP problem with given parameters
def gurobi_SSKP(sample_k, r, c, q):
    # input:
    # sample_k (weight matrix): 2d numpy array of shape (k, m), where m is the number of items, k is the number of samples
    # r (reward vector): 1d numpy array of shape (m, )
    # c (cost vector): scalar
    # q (budget): scalar
    # output: 
    # x (selection vector): 1d numpy array of shape (m, )
    k = len(sample_k)
    m = len(sample_k[0])
    env = Env(empty=True)
    env.setParam(GRB.Param.OutputFlag, 0)
    env.start()
    model = Model("SSKP", env=env)
    # model.setParam(GRB.Param.OutputFlag, 0) # suppress gurobi output
    model.setParam(GRB.Param.MIPGap, 0.00001)
    # model.setParam(GRB.Param.LogToConsole, 0)
    x = model.addVars(m, vtype=GRB.BINARY, name="x")
    z = model.addVars(k, lb=0, vtype=GRB.CONTINUOUS, name="z")

    model.setObjective(quicksum(r[i] * x[i] for i in range(m)) - c/k * quicksum(z[j] for j in range(k)), GRB.MAXIMIZE)
    model.addConstrs((z[j] >= quicksum(sample_k[j][i] * x[i] for i in range(m)) - q for j in range(k)), name="budget")
    model.setParam(GRB.Param.Threads, 1)

    model.optimize()

    if model.status == GRB.OPTIMAL:
        return tuple(int(x[i].X) for i in range(m))
    else:
        print("No optimal solution found.")
        return None
    

def gurobi_portfolio(sample_xi, p, mu, b):
    # input:
    # sample_xi(random returns): k*m numpy array (where m is number of assets)
    # p(price): m*1 numpy array
    # mu(mean return): m*1 numpy array
    # b(budget): 1*1
    # output:
    # x(number of shares for each asset): m*1 numpy array

    k = len(sample_xi)
    m = len(sample_xi[0])
    env = Env(empty=True)
    env.setParam(GRB.Param.OutputFlag, 0)
    env.start()
    model = Model("portfolio", env=env)
    # model.setParam(GRB.Param.OutputFlag, 0)
    model.setParam(GRB.Param.MIPGap, 0.00001)
    # model.setParam(GRB.Param.LogToConsole, 0)
    x = model.addVars(m, lb=0, vtype=GRB.INTEGER, name="x")
    c = model.addVar(vtype=GRB.CONTINUOUS, name="c")
    z = model.addVars(k, lb=0, vtype=GRB.CONTINUOUS, name="z")
    
    model.setObjective(c + 1/0.05 * 1/k * quicksum(z[i] for i in range(k)), GRB.MINIMIZE)
    
    model.addConstrs(z[i] >= -sum(sample_xi[i][j] * x[j] * p[j] for j in range(m)) - c for i in range(k))
    model.addConstr(quicksum(mu[j] * x[j] * p[j] for j in range(m)) >= 1.5*b)
    model.addConstr(quicksum(x[j] * p[j] for j in range(m)) <= b)
    model.setParam(GRB.Param.Threads, 1)

    model.optimize()

    if model.status == GRB.OPTIMAL:
        return tuple(int(x[j].X) for j in range(m))
    else:
        print("No optimal solution found.")
        return None