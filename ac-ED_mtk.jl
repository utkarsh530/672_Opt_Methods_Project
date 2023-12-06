using Pkg
Pkg.activate(".")

using PowerModels
using Ipopt
using JuMP
import ModelingToolkit: ≲
# Load System Data
# ----------------
powermodels_path = joinpath(dirname(pathof(PowerModels)), "..")
#file_name = "$(powermodels_path)/test/data/matpower/case69.m"
file_name = joinpath(@__DIR__, "Data/case57.m")
# note: change this string to modify the network data that will be loaded

# load the data file
data = PowerModels.parse_file(file_name)

# Add zeros to turn linear objective functions into quadratic ones
# so that additional parameter checks are not required
PowerModels.standardize_cost_terms!(data, order=2)

# Adds reasonable rate_a values to branches without them
PowerModels.calc_thermal_limits!(data)

# use build_ref to filter out inactive components
ref = PowerModels.build_ref(data)[:it][:pm][:nw][0]
# note: ref contains all the relevant system parameters needed to build the OPF model
# When we introduce constraints and variable bounds below, we use the parameters in ref.

###############################################################################
# 2. Economic Dispatch (ED) Model
###############################################################################

# Define the demand at each bus
demand = Dict{Int, Float64}(i => (isempty(ref[:bus_loads][i]) ? 0.0 : sum(ref[:load][l]["pd"] for l in ref[:bus_loads][i])) for (i, _) in ref[:bus])

using ModelingToolkit

vars = Num[]

lb = Float64[]
ub = Float64[]

## Add Variables
## JuMP code:
# @variable(model, ref[:gen][i]["pmin"] <= pg_ed[i in keys(ref[:gen])] <= ref[:gen][i]["pmax"])

ModelingToolkit.@variables pg_ed[1:maximum(keys(ref[:gen]))]

for i in keys(ref[:gen])
    push!(lb, ref[:gen][i]["pmin"])
    push!(ub, ref[:gen][i]["pmax"])
end

# Economic Dispatch Objective Function
# -------------------------------------

# Minimize the total generation cost

# @objective(model, Min, sum(ref[:gen][i]["cost"][1] * pg_ed[i]^2 + ref[:gen][i]["cost"][2] * pg_ed[i] + ref[:gen][i]["cost"][3] for (i, _) in ref[:gen]))
loss = sum(gen["cost"][1] * pg_ed[i]^2 + gen["cost"][2] * pg_ed[i] + gen["cost"][3] for (i, gen) in ref[:gen])

# Economic Dispatch Constraints
# ------------------------------

# Nodal power balance constraints
# for (i, bus) in ref[:bus]
#     bus_loads = [ref[:load][l] for l in ref[:bus_loads][i]]
#     bus_shunts = [ref[:shunt][s] for s in ref[:bus_shunts][i]]

#    # @constraint(model,
#    #     sum(pg_ed[g] for g in ref[:bus_gens][i]) - sum(load["pd"] for load in bus_loads) >=0
#     #)
#     # Calculate the total generation
# total_generation = sum(pg_ed[g] for (i, bus_gens) in ref[:bus_gens] for g in bus_gens)

# # Add constraint for total load equaling total generation
# @constraint(model, total_generation >= sum(demand[i] for (i, _) in ref[:bus]))

# end

cons = Array{Union{ModelingToolkit.Equation,ModelingToolkit.Inequality}}([])

total_generation = sum(pg_ed[g] for (i, bus_gens) in ref[:bus_gens] for g in bus_gens)

# Add constraint for total load equaling total generation
push!(cons, sum(demand[i] for (i, _) in ref[:bus]) - total_generation  ≲ 0)


vars = vcat(vars, [pg_ed[i] for i in keys(ref[:gen])])

optsys = ModelingToolkit.OptimizationSystem(loss, vars, [], constraints=cons, name=:rosetta)

u0map = Dict([k => 0.0 for k in collect(optsys.states)])

inds = Int[]
for k in collect(optsys.states)
    push!(inds, findall(x -> isequal(x, k), vars)[1])
end

using Optimization
prob = Optimization.OptimizationProblem(optsys, u0map, lb = lb[inds], ub = ub[inds], grad=true, hess=true, cons_j=true, cons_h=true, cons_sparse=true, sparse=true)


using OptimizationMOI

PRINT_LEVEL = 0

opt_sol = OptimizationMOI.solve(prob, Ipopt.Optimizer())

using PSOGPU
n_particles = 500


Optimization.SciMLBase.allowsconstraints(ParallelPSOKernel) = true

test_sol = solve(prob, ParallelPSOKernel(n_particles; threaded = true), maxiters = 500)
