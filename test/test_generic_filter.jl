using Test
using GeometricFilter
using Manifolds


using PDMats

using Random

rng = Random.default_rng()

"""
Zero stochastic motion
"""
function setup_static_motion(action, dev=0)
    motion = ZeroMotion(action)
    pnoise = ActionNoise(action, dev)
    return StochasticMotion(motion, pnoise)
end

"""
Identity observer
"""
function setup_id_observer(action, dev=1.0)
    M = group_manifold(action)
    obs = IdentityObserver(M) # or specified?
    onoise = IsotropicNoise(observation_space(obs), dev)
    return obs, onoise
end

"""
Distribution from random point.
"""
function setup_initial_dist(rng, pnoise, dev=1.0)
    x0 = rand(rng, sample_space(pnoise))
    D0 = ActionDistribution(x0, update_cov(pnoise, dev^2))
    return D0
end


function simple_params(rng, action, pdev=1, odev=1, ddev=1)
    sm = setup_static_motion(action, pdev)
    obs, onoise = setup_id_observer(action, odev)
    D0 = setup_initial_dist(rng, sm.noise, ddev)
    return D0, sm, obs, onoise
end

function run_filter(rng, D0, sm, obs, onoise)
    x1 = simulate(sm, D0.μ, PositionPerturbation(rng))
    y1 = onoise(rng, obs(x1))

    # motion_ = noisy_motion(A, rng, x1)
    # smotion_ = sensor_perturbation(rng, sm, x1)

    D1 = predict(D0, sm)

    D1_ = update(D1, Observation(obs, onoise, y1))

    return x1, D1,D1_
end

param_list = [
    simple_params(rng, GroupOperationAction(MultiDisplacementGroup(2,1))),
    simple_params(rng, DualGroupOperationAction(MultiDisplacementGroup(2))),
    let
        G = MultiDisplacementGroup(2)
        OA = MultiAffineAction(G, RightAction())
        landmarks = [randn(rng, manifold_dimension(group_manifold(OA))) for k in 1:5]
        obs = ProductObserver([ActionObserver(OA, ref) for ref in landmarks]...)
        onoise = IsotropicNoise(observation_space(obs), 1.)
        action = GroupOperationAction(G)
        pnoise = ActionNoise(action, 1.)
        D0 = setup_initial_dist(rng, pnoise)
        sm = setup_static_motion(action)
        D0, sm, obs, onoise
    end,
]


@testset "Generic Filter" for params in param_list
    D0, sm, obs, onoise = params
    M = group_manifold(ManifoldNormal.get_action(D0))
    x1, D1,D1_ = run_filter(rng, D0, sm, obs, onoise)

end

