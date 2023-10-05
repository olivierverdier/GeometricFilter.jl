module GeometricFilter

# MultiAffine
export MultiAffine,
    MultiDisplacement,
    from_normal_grp, from_normal_alg,
    to_factor_grp, to_factor_alg

export MultiAffineAction

# export get_id_matrix_lie


# Motion
export AbstractMotion, AbstractAffineMotion,
    AffineMotion, RigidMotion, TranslationMotion,
    MultiAffineMotion, FlatAffineMotion,
    ZeroMotion,
    get_flat_action,
    # integrate,
    # compose_adjoint,
    swap_group_motion,
    get_action

# Simulation
export integrate, generate_signal,
    AbstractStochasticMotion, StochasticMotion,
    DataMode,
    SensorPerturbation, PositionPerturbation,
    simulate_observations,
    get_noisy_observations,
    noisy_observation,
    sensor_perturbation,
    simulate_filter

# Filter
export predict,
    add_process_noise,
    update

# Observation
export AbstractObserver,
    ProductObserver,
    ActionObserver,
    PositionObserver,
    LinearObserver,
    IdentityObserver,
    observation_space, observed_space
    # get_tan_observer,
    # get_manifold,
    # get_action
export Observation, EmptyObservation

export ConstantFunction

# Noise
export AbstractNoise,
    ActionNoise, IsotropicNoise,
    NoNoise,
    sample_space,
    get_basis_at, get_covariance_at,
    update_cov

export Covariance, covariance_from
export AbstractProjLogNormal, ProjLogNormal,
    action_noise, scaled_distance,
    update_mean_cov, update_mean

export DualGroupOperationAction

export inverse_adjoint_action

using Manifolds
import ManifoldsBase

using GeometricFilter

import Random
import Distributions
import PDMats

import LinearAlgebra

import ManifoldDiffEq
import OrdinaryDiffEq

import DataFrames


include("Aux/Manifolds/MultiAffine.jl")
include("Aux/Manifolds/MultiDisplacement.jl")
include("Aux/Manifolds/rotation_action.jl")

include("Aux/Manifolds/DualGroupOperationAction.jl")

include("Utils.jl")

include("GroupTools.jl")

include("Motion.jl")

include("Aux/PDMats/Covariance.jl")
include("ProjLogNormal.jl")
include("ConstantFunction.jl")
include("Noise.jl")

include("Observation.jl")
include("StochasticMotion.jl")

include("Filter/Prediction.jl")
include("Filter/Update.jl")
include("Filter/Simulation.jl")

include("Aux/Manifolds/MultiAffineAction.jl")

include("Observer.jl")





end
