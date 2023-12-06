
import PowerModels
import Ipopt
using ModelingToolkit, Optimization, OptimizationMOI
import ModelingToolkit: ≲


# Load System Data
# ----------------
powermodels_path = joinpath(dirname(pathof(PowerModels)), "..")
#file_name = "$(powermodels_path)/test/data/matpower/case5.m"
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

vars = Num[]
lb = Float64[]
ub = Float64[]

ModelingToolkit.@variables va[1:maximum(keys(ref[:bus]))]
for i in keys(ref[:bus])
    push!(lb, -Inf)
    push!(ub, Inf)
end
ModelingToolkit.@variables vm[1:maximum(keys(ref[:bus]))]
for i in keys(ref[:bus])
    push!(lb, ref[:bus][i]["vmin"])
    push!(ub, ref[:bus][i]["vmax"])
end
vars = vcat(vars, [va[i] for i in keys(ref[:bus])], [vm[i] for i in keys(ref[:bus])])
ModelingToolkit.@variables pg[1:maximum(keys(ref[:gen]))]
for i in keys(ref[:gen])
    push!(lb, ref[:gen][i]["pmin"])
    push!(ub, ref[:gen][i]["pmax"])
end
ModelingToolkit.@variables qg[1:maximum(keys(ref[:gen]))]
for i in keys(ref[:gen])
    push!(lb, ref[:gen][i]["qmin"])
    push!(ub, ref[:gen][i]["qmax"])
end
vars = vcat(vars, [pg[i] for i in keys(ref[:gen])], [qg[i] for i in keys(ref[:gen])])
i_inds, j_inds, l_inds = maximum(first.(ref[:arcs])), maximum(getindex.(ref[:arcs], Ref(2))), maximum(last.(ref[:arcs]))
ModelingToolkit.@variables p[1:i_inds, 1:j_inds, 1:l_inds]
ModelingToolkit.@variables q[1:i_inds, 1:j_inds, 1:l_inds]

for (l, i, j) in ref[:arcs]
    push!(vars, p[l, i, j])
    push!(lb, -ref[:branch][l]["rate_a"])
    push!(ub, ref[:branch][l]["rate_a"])
end

for (l, i, j) in ref[:arcs]
    push!(vars, q[l, i, j])
    push!(lb, -ref[:branch][l]["rate_a"])
    push!(ub, ref[:branch][l]["rate_a"])
end

loss = sum(gen["cost"][1] * pg[i]^2 + gen["cost"][2] * pg[i] + gen["cost"][3] for (i, gen) in ref[:gen])

cons = Array{Union{ModelingToolkit.Equation,ModelingToolkit.Inequality}}([])
for (i, bus) in ref[:ref_buses]
    push!(cons, va[i] ~ 0)
end

for (i, bus) in ref[:bus]
    bus_loads = [ref[:load][l] for l in ref[:bus_loads][i]]
    bus_shunts = [ref[:shunt][s] for s in ref[:bus_shunts][i]]
    push!(cons,
        sum(p[a...] for a in ref[:bus_arcs][i]) ~
            (sum(pg[g] for g in ref[:bus_gens][i]; init = 0.0)) -
            (sum(load["pd"] for load in bus_loads; init = 0.0)) -
            sum(shunt["gs"] for shunt in bus_shunts; init = 0.0)*vm[i]^2
    )

    push!(cons,
        sum(q[a...] for a in ref[:bus_arcs][i]) ~
            (sum(qg[g] for g in ref[:bus_gens][i]; init = 0.0)) -
            (sum(load["qd"] for load in bus_loads; init = 0.0))
            + sum(shunt["bs"] for shunt in bus_shunts; init = 0.0)*vm[i]^2
    )
end

# Branch power flow physics and limit constraints
for (i, branch) in ref[:branch]
    f_idx = (i, branch["f_bus"], branch["t_bus"])
    t_idx = (i, branch["t_bus"], branch["f_bus"])

    p_fr = p[f_idx...]
    q_fr = q[f_idx...]
    p_to = p[t_idx...]
    q_to = q[t_idx...]

    vm_fr = vm[branch["f_bus"]]
    vm_to = vm[branch["t_bus"]]
    va_fr = va[branch["f_bus"]]
    va_to = va[branch["t_bus"]]

    g, b = PowerModels.calc_branch_y(branch)
    tr, ti = PowerModels.calc_branch_t(branch)
    ttm = tr^2 + ti^2
    g_fr = branch["g_fr"]
    b_fr = branch["b_fr"]
    g_to = branch["g_to"]
    b_to = branch["b_to"]

    # From side of the branch flow
    push!(cons, p_fr ~ (g + g_fr) / ttm * vm_fr^2 + (-g * tr + b * ti) / ttm * (vm_fr * vm_to * cos(va_fr - va_to)) + (-b * tr - g * ti) / ttm * (vm_fr * vm_to * sin(va_fr - va_to)))
    push!(cons, q_fr ~ -(b + b_fr) / ttm * vm_fr^2 - (-b * tr - g * ti) / ttm * (vm_fr * vm_to * cos(va_fr - va_to)) + (-g * tr + b * ti) / ttm * (vm_fr * vm_to * sin(va_fr - va_to)))

    # To side of the branch flow
    push!(cons, p_to ~ (g + g_to) * vm_to^2 + (-g * tr - b * ti) / ttm * (vm_to * vm_fr * cos(va_to - va_fr)) + (-b * tr + g * ti) / ttm * (vm_to * vm_fr * sin(va_to - va_fr)))
    push!(cons, q_to ~ -(b + b_to) * vm_to^2 - (-b * tr + g * ti) / ttm * (vm_to * vm_fr * cos(va_to - va_fr)) + (-g * tr - b * ti) / ttm * (vm_to * vm_fr * sin(va_to - va_fr)))

    # Voltage angle difference limit
    push!(cons, va_fr - va_to ≲ branch["angmax"])
    push!(cons, branch["angmin"] ≲ va_fr - va_to)

    # Apparent power limit, from side and to side
    push!(cons, p_fr^2 + q_fr^2 ≲ branch["rate_a"]^2)
    push!(cons, p_to^2 + q_to^2 ≲ branch["rate_a"]^2)
end

optsys = ModelingToolkit.OptimizationSystem(loss, vars, [], constraints=cons, name=:rosetta)

optsys = ModelingToolkit.structural_simplify(optsys)

u0map = Dict([k => 0.0 for k in collect(optsys.states)])
for key in keys(ref[:bus])
    if vm[key] in keys(u0map)
        # println(vm[key])
        u0map[vm[key]] = 1.0
    end
end

inds = Int[]
for k in collect(optsys.states)
    push!(inds, findall(x -> isequal(x, k), vars)[1])
end
prob = Optimization.OptimizationProblem(optsys, u0map, lb = lb[inds], ub = ub[inds], grad=true, hess=true, cons_j=true, cons_h=true, cons_sparse=true, sparse=true)

solve_time_with_compilation = @elapsed opt_sol = OptimizationMOI.solve(prob, Ipopt.Optimizer(); print_level = 0)
solve_time_without_compilation = @elapsed opt_sol = OptimizationMOI.solve(prob, Ipopt.Optimizer(); print_level = 0)

dataset = (;data, ref)
mtk_test_res = solve_opf_mtk(dataset)
