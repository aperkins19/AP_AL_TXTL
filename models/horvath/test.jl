using Sundials
using DifferentialEquations
using DataFrames

f(u,p,t) = 1.01*u
u0 = 1/2
tspan = (0.0,1.0)
prob = ODEProblem(f,u0,tspan)
sol = solve(prob, CVODE_BDF(), reltol=1e-8, abstol=1e-8)
df = DataFrame(sol);
print(df)


@info "test complete"