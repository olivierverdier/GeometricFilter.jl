using Test
using GeometricFilter
using Manifolds

import LinearAlgebra

import PDMats

import Random

"""
Simple test: assume that the movement is zero, process noise zero.
We use the same measurement over and over again.
The point should then converge to the true value.
"""


rng = Random.default_rng()



function setup_initial_dist(rng, A, dev=1.)
    x0 = rand(rng, group_manifold(A))
    D0 = ProjLogNormal(A, x0, dev^2)
    return D0
end

function setup_action_observer(rng, OA, N=10)
    refs = [rand(rng, group_manifold(OA)) for i in 1:N]
    obs = ProductObserver([ActionObserver(OA, ref) for ref in refs]...)
    return obs
end


run_filter(D0, N, observer, onoise, measure) =
    accumulate(1:N; init=D0) do D, i
        return update(D, Observation(observer, onoise, measure))
    end

@testset "Observer action $OA\nAction $A\nObserver $(typeof(observer)):" for OA in [
    GroupOperationAction(SpecialOrthogonal(3)),
    DualGroupOperationAction(SpecialOrthogonal(3)),
    MultiAffineAction(MultiDisplacement(2), LeftAction()),
    MultiAffineAction(MultiDisplacement(2), RightAction()),
    RotationAction(Euclidean(3), SpecialOrthogonal(3), RightAction()),
    ], A in [
        GroupOperationAction(base_group(OA)),
        DualGroupOperationAction(base_group(OA)),
    ], observer in
    [
        setup_action_observer(rng, OA, 10),
        IdentityObserver(base_group(A)),
    ]

    M = group_manifold(A)

    # ground truth
    D0 = setup_initial_dist(rng, A)

    x0 = D0.μ


    m = observer(x0)

    onoise = IsotropicNoise(observation_space(observer), 1.0)


    # actually run the test
    noise = ActionNoise(A, 1.0)
    x1 = noise(rng, x0)
    D1 = update_mean(D0, x1)

    Ds = run_filter(D1, 2^5, observer, onoise, m)

    # gdist = [distance(group_manifold(A), D0.μ, D.μ) for D in Ds]
    dists = [distance(group_manifold(A), D0.μ, D.μ) for D in [first(Ds), last(Ds)]]

    improvement_dB = -10*log10(last(dists) / first(dists))

    @test improvement_dB >= 10
end

# why is the error not exponential as in the localisation test?!?

