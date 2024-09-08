module GeometricFilter

using AffineMotions
using ManifoldNormal

import RecursiveArrayTools

# Simulation
export simulate, generate_signal,
    AbstractStochasticMotion, StochasticMotion,
    DataMode,
    SensorPerturbation, PositionPerturbation,
    noisy_observation,
    simulate_filter,
    simulate_observations,
    get_noisy_observations,
    sensor_perturbation

# Filter
export predict,
    update

# Observation
export AbstractObserver,
    ProductObserver,
    ActionObserver,
    LinearObserver,
    IdentityObserver,
    observation_space, observed_space,
    get_action

export Observation, EmptyObservation




export DualGroupOperationAction


using Manifolds



import Random
import Distributions
import PDMats
import ManifoldGroupUtils as GU
import SparseArrays: SparseVector

import LinearAlgebra



DualGroupOperationAction(G) = GroupOperationAction(G, Manifolds.LeftBackwardAction())


include("Utils.jl")


include("Observation.jl")
include("Filter/SimulationMode.jl")
include("Filter/StochasticMotion.jl")

include("Filter/Prediction.jl")
include("Filter/Update.jl")
include("Filter/Simulation.jl")


include("Observer.jl")


export PositionObserver

using MultiAffine
PositionObserver(A::MultiAffineAction{LeftAction, <:MultiAffineGroup{<:Any, dim}}) where {dim} = ActionObserver(A, zeros(dim))


end
