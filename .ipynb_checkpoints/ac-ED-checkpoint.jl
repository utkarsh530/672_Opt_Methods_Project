#### AC Optimal Power Flow ####

# This file provides a pedagogical example of modeling the AC Optimal Power
# Flow problem using the Julia Mathematical Programming package (JuMP) and the
# PowerModels package for data parsing.

# This file can be run by calling `include("ac-opf.jl")` from the Julia REPL or
# by calling `julia ac-opf.jl` in Julia v1.

# Developed by Line Roald (@lroald) and Carleton Coffrin (@ccoffrin)


###############################################################################
# 0. Initialization
###############################################################################

# Load Julia Packages
#--------------------
using PowerModels
using Ipopt
using JuMP


# Load System Data
# ----------------
powermodels_path = joinpath(dirname(pathof(PowerModels)), "..")
#file_name = "$(powermodels_path)/test/data/matpower/case69.m"
file_name = "C:/Users/mansi/Dropbox/6.7201_optimization_methods/Project/Data/case57.m"
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
# 1. Building the Optimal Power Flow Model
###############################################################################

# Initialize a JuMP Optimization Model
#-------------------------------------
model = Model(Ipopt.Optimizer)

set_optimizer_attribute(model, "print_level", 0)
# note: print_level changes the amount of solver information printed to the terminal


###############################################################################
# 2. Economic Dispatch (ED) Model
###############################################################################

# Define the demand at each bus
demand = Dict{Int, Float64}(i => (isempty(ref[:bus_loads][i]) ? 0.0 : sum(ref[:load][l]["pd"] for l in ref[:bus_loads][i])) for (i, _) in ref[:bus])


# Add Economic Dispatch Variables
# --------------------------------

# Add active power generation variable pg_ed for each generator (including limits)
@variable(model, ref[:gen][i]["pmin"] <= pg_ed[i in keys(ref[:gen])] <= ref[:gen][i]["pmax"])

# Economic Dispatch Objective Function
# -------------------------------------

# Minimize the total generation cost
@objective(model, Min, sum(ref[:gen][i]["cost"][1] * pg_ed[i]^2 + ref[:gen][i]["cost"][2] * pg_ed[i] + ref[:gen][i]["cost"][3] for (i, _) in ref[:gen]))

# Economic Dispatch Constraints
# ------------------------------

# Nodal power balance constraints
for (i, bus) in ref[:bus]
    bus_loads = [ref[:load][l] for l in ref[:bus_loads][i]]
    bus_shunts = [ref[:shunt][s] for s in ref[:bus_shunts][i]]

   # @constraint(model,
   #     sum(pg_ed[g] for g in ref[:bus_gens][i]) - sum(load["pd"] for load in bus_loads) >=0
    #)
    # Calculate the total generation
total_generation = sum(pg_ed[g] for (i, bus_gens) in ref[:bus_gens] for g in bus_gens)

# Add constraint for total load equaling total generation
@constraint(model, total_generation >= sum(demand[i] for (i, _) in ref[:bus]))

end


# Solve the Economic Dispatch Model and measure the time
@time begin
    optimize!(model)
end
# Check the value of the objective function
cost = objective_value(model)
println("The cost of generation is $(cost).")
# Check the results
println("Economic Dispatch Results:")
println("The solver termination status is $(termination_status(model))")

# ...

# Display the Economic Dispatch results
for (i, gen) in ref[:gen]
    pg_ed_result = value(pg_ed[i]) * ref[:baseMVA]
    println("Generator $i: Active Power Generation = $pg_ed_result MW")
end

