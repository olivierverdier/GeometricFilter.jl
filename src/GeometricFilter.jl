module GeometricFilter

using AffineMotions

# Simulation
export integrate, generate_signal,
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
    add_process_noise,
    update

# Observation
export AbstractObserver,
    ProductObserver,
    ActionObserver,
    LinearObserver,
    IdentityObserver,
    observation_space, observed_space

export Observation, EmptyObservation


# Noise
export AbstractNoise,
    ActionNoise, IsotropicNoise,
    NoNoise,
    sample_space,
    get_covariance_at,
    update_cov

export AbstractProjLogNormal, ProjLogNormal,
    action_noise, scaled_distance,
    update_mean_cov, update_mean

export get_action

export DualGroupOperationAction


import ManifoldsBase
using Manifolds


using ConstantFunctions

import Random
import Distributions
import PDMats
import PDMatsSingular: sample
import ManifoldGroupUtils as GU

import LinearAlgebra



DualGroupOperationAction(G) = GroupOperationAction(G, Manifolds.LeftBackwardAction())
_get_group_operation_action(G, ::LeftAction) = GroupOperationAction(G, (LeftAction(), LeftSide()))
_get_group_operation_action(G, ::RightAction) = DualGroupOperationAction(G)


include("Utils.jl")

include("ProjLogNormal.jl")
include("Noise.jl")

include("Observation.jl")
include("Filter/SimulationMode.jl")
include("StochasticMotion.jl")

include("Filter/Prediction.jl")
include("Filter/Update.jl")
include("Filter/Simulation.jl")


include("Observer.jl")


export PositionObserver

using MultiAffine
PositionObserver(A::MultiAffineAction{LeftAction, <:MultiAffineGroup{<:Any, dim}}) where {dim} = ActionObserver(A, zeros(dim))


end
