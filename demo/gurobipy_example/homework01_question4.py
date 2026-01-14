import gurobipy as gp
from gurobipy import GRB

m = gp.Model("lp1")

x1 = m.addVar(name="x1", vtype=GRB.INTEGER)
x2 = m.addVar(name="x2", vtype=GRB.INTEGER)

m.setObjective(20 * x1 + 15 * x2, GRB.MINIMIZE)

m.addConstr(0.3 * x1 + 0.4 * x2 >= 2000, "c1")
m.addConstr(0.4 * x1 + 0.2 * x2 >= 1500, "c2")
m.addConstr(0.2 * x1 + 0.3 * x2 >= 500, "c3")
m.addConstr(x1 >= 0, "c4")
m.addConstr(x2 >= 0, "c5")
m.addConstr(x1 <= 9000, "c6")
m.addConstr(x2 <= 6000, "c7")

m.optimize()

if m.status == GRB.OPTIMAL:
    print('Optimal solution found')
else:
    print('Optimal solution NOT found')

for v in m.getVars():
    print(f'{v.VarName} {v.x}')

print(f"Obj: {m.objVal}")
