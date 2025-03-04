import gurobipy as gp
from gurobipy import GRB

m = gp.Model("lp1")

x1 = m.addVar(name="x1")
x2 = m.addVar(name="x2")

m.setObjective(2 * x1 + 3 * x2, GRB.MAXIMIZE)

m.addConstr(x1 + 4 * x2 <= 8, "c1")
m.addConstr(x1 - x2 <= 1, "c2")
m.addConstr(x1 >= 0, "c3")
m.addConstr(x2 >= 0, "c4")

m.optimize()

if m.status == GRB.OPTIMAL:
    print('Optimal solution found')
else:
    print('Optimal solution NOT found')

for v in m.getVars():
    print(f'{v.VarName} {v.x}')

print(f"Obj: {m.objVal}")
