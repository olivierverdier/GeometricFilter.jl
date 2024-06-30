module GeometricFilter


# Motion
export AbstractMotion, AbstractAffineMotion,
    RigidMotion, TranslationMotion,
    AdjointLinearMotion,
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
    PositionObserver,
    LinearObserver,
    IdentityObserver,
    observation_space, observed_space

export Observation, EmptyObservation

export ConstantFunction

# Noise
export AbstractNoise,
    ActionNoise, IsotropicNoise,
    NoNoise,
    sample_space,
    get_covariance_at,
    update_cov

export Covariance, covariance_from
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

# weak dependency: only used in AdjointLinearMotion and PositionObserver
import MultiAffine:
    MultiAffineGroup, MultiAffineAction,
    from_normal_alg, to_factor_grp
import GroupTools

using GeometricFilter

import Random
import Distributions
import PDMats

import LinearAlgebra

import ManifoldDiffEq
import OrdinaryDiffEq


DualGroupOperationAction(G) = GroupOperationAction(G, Manifolds.LeftBackwardAction())
_get_group_operation_action(G, ::LeftAction) = GroupOperationAction(G, (LeftAction(), LeftSide()))
_get_group_operation_action(G, ::RightAction) = DualGroupOperationAction(G)


include("Utils.jl")


include("Motion.jl")

include("Aux/PDMats/Covariance.jl")
include("ProjLogNormal.jl")
include("ConstantFunction.jl")
include("Noise.jl")

include("Observation.jl")
include("Filter/SimulationMode.jl")
include("StochasticMotion.jl")

include("Filter/Prediction.jl")
include("Filter/Update.jl")
include("Filter/Simulation.jl")


include("Observer.jl")





end
