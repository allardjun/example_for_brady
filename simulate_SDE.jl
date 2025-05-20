using DifferentialEquations, DiffEqCallbacks, Random, LinearAlgebra, Statistics
using JLD2
using ComponentArrays

Random.seed!(2025)

# For processor/core info
println("Threads.nthreads() = ", Threads.nthreads())   # Number of available threads
println("Sys.CPU_THREADS = ", Sys.CPU_THREADS)   # Logical cores (threads)
println("Sys.CPU_NAME  = ", Sys.CPU_NAME)       # CPU model name

# Define constants
D = 1.0
eps = 1.0
R = 10.0 
t_end = 2.0 # overridden in sweep

sigma = sqrt(2*D)


function f!(du,u,p,t)   # drift
    du .= (p.u0 - u)./ (p.t_end - t)  # drift term of the 3-D Brownian bridge
    # du .= sin.(t) .- 0.2*u
end
g!(du,u,p,t) = fill!(du, sigma)   # diagonal noise

u0     = [R+eps, 0.0, 0.0]

is_inside(u,t,integrator) = R - norm(u)
stopper! = (integ) -> SciMLBase.terminate!(integ)
hit_cb = ContinuousCallback(is_inside, stopper!; save_positions=(true,false))

function simulate_ensemble(N::Int; t_end=t_end)

    tspan  = (0.0, t_end-1e-7)

    p = ComponentArray(
        u0 = u0,
        t_end = t_end,
    )

    prob  = SDEProblem(f!, g!, u0, tspan, p; callback=hit_cb)

    ens   = EnsembleProblem(prob)

    sols  = solve(ens, SOSRI(); trajectories=N, reltol=1e-4, abstol=1e-4,
                # parallel=:threads, 
                dtmax=0.005)

    S = 1 - mean(s -> s.retcode == SciMLBase.ReturnCode.Terminated, sols)
    @show t_end, S

    return S

end

# simulate_ensemble(1000; t_end=t_end)


t_values = 10 .^ range(log10(0.1), log10(1000), length=4)

N = 10 # was 10_000 for free

@time S_SDE_values = map(t -> simulate_ensemble(N; t_end=t), t_values)

@save "S_handcuffed_SDE_2505071200.jld2" t_values S_SDE_values

##

# Filter strictly positive values of S_SDE_values
filtered_data = [(t, S) for (t, S) in zip(t_values, S_SDE_values) if S > 0]
filtered_t_values = [t for (t, _) in filtered_data]
filtered_S_SDE_values = [S for (_, S) in filtered_data]

# ---------------------------------------------------
## This is a cell delimiter

# Plot the filtered data
using CairoMakie
fig = Makie.Figure()
ax = Makie.Axis(fig[1, 1])
ax.xlabel = "t (total length of 2 walks)"
ax.ylabel = "Survival probability"


scatter!(ax, filtered_t_values, filtered_S_SDE_values, 
    label="S SDE", 
    color=:gray, 
    transparency=true, 
    alpha=0.5,  # Set transparency level (0.0 is fully transparent, 1.0 is opaque)
    overdraw=true  # Render behind other elements
)

ax.xscale = Makie.log10
ax.yscale = Makie.log10

display(fig)

save("filtered_SDE.pdf", fig)
