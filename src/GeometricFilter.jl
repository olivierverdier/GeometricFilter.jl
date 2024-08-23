module GeometricFilter


# Motion
export AbstractMotion, AbstractAffineMotion,
    RigidMotion, TranslationMotion,
    FlatAffineMotion,
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

export DualGroupOperationAction


import ManifoldsBase
import Manifolds
import ManifoldsBase:  # general manifolds
    AbstractManifold, manifold_dimension,
    AbstractBasis, CachedBasis, DefaultOrthonormalBasis, zero_vector, get_coordinates, get_vector,
    allocate_result,
    TangentSpace, is_point,
    submanifold_component, submanifold_components, submanifold
import Manifolds:
    ArrayPartition, ProductManifold
import Manifolds: # Groups
    Identity, identity_element, TranslationGroup, Euclidean, lie_bracket, exp_lie, translate_diff, get_vector_lie
import Manifolds: # Actions
    AbstractGroupAction, apply, apply!, adjoint_action, TranslationAction, LeftAction, RightAction, switch_direction, ActionDirection, GroupOperationAction, LeftSide, RightSide, base_group, group_manifold, apply_diff_group

import ManifoldGroupUtils as GU

using ConstantFunctions

import Random
import Distributions
import PDMats
import PDMatsSingular: sample

import LinearAlgebra

import ManifoldDiffEq
import OrdinaryDiffEq


DualGroupOperationAction(G) = GroupOperationAction(G, Manifolds.LeftBackwardAction())
_get_group_operation_action(G, ::LeftAction) = GroupOperationAction(G, (LeftAction(), LeftSide()))
_get_group_operation_action(G, ::RightAction) = DualGroupOperationAction(G)


include("Utils.jl")

include("Motion.jl")

include("ProjLogNormal.jl")
include("Noise.jl")

include("Observation.jl")
include("Filter/SimulationMode.jl")
include("StochasticMotion.jl")

include("Filter/Prediction.jl")
include("Filter/Update.jl")
include("Filter/Simulation.jl")


include("Observer.jl")

include("../ext/FilterMultiAffineExt/FilterMultiAffineExt.jl")



end
