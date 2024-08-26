using Test
using GeometricFilter
import Manifolds: SpecialOrthogonal, GroupOperationAction, LeftAction, RightAction, Euclidean, RotationAction, base_group, group_manifold, distance

import LinearAlgebra
using Distributions

import PDMats

import Random

"""
Simple test:
Pick a group, and choose either the natural or dual left action.
Assume that the movement is zero, process noise zero.
The observer may be the identity, or an action observation
(so we observe ``χ ⋅ x_0``, ``χ ⋅ x_1``, for randomly chosen points ``x_i``).
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


run_filter(D0, N, observation) =
    foldl(1:N; init=D0) do D, i
        return update(D, observation)
    end

dual_display(n) = ["", "*"][n]
@testset "$ioa$(dual_display(dual))  $ia $io" for (ioa,OAL) in ([
    ("SO", [GroupOperationAction(SpecialOrthogonal(3)),
            DualGroupOperationAction(SpecialOrthogonal(3))]),
    ("Mv", [MultiAffineAction(MultiDisplacement(2), LeftAction()),
            MultiAffineAction(MultiDisplacement(2), RightAction())]),
    ("Rv", [RotationAction(Euclidean(3), SpecialOrthogonal(3), RightAction())]),
    ]),
    (dual, OA) in enumerate(OAL),
    (ia,A) in ([
        ("G", GroupOperationAction(base_group(OA))),
        ("G*", DualGroupOperationAction(base_group(OA))),
    ]), io in
    ([
        :ao10,
        :Id
    ])

    # with the Identity observer, the observation action is not used,
    # so run one test instead of another, identical one:
    if io == "Id" && dual == 2
        continue
    end

    Random.seed!(rng, 10)

    if io == :ao10
        observer = setup_action_observer(rng, OA, 10)
    elseif io == :Id
        observer = IdentityObserver(base_group(A))
    end

    M = group_manifold(A)

    # ground truth
    D0 = setup_initial_dist(rng, A)

    x0 = mean(D0)


    m = observer(x0)

    onoise = IsotropicNoise(observation_space(observer), 1.0)


    # actually run the test
    noise = ActionNoise(A, 1.0)
    x1 = noise(rng, x0)
    D1 = update_mean(D0, x1)

    D_ = run_filter(D1, 2^6, Observation(observer, onoise, m))

    # gdist = [distance(group_manifold(A), D0.μ, D.μ) for D in Ds]
    # dists = [distance(group_manifold(A), D0.μ, D.μ) for D in [first(Ds), last(Ds)]]
    dists = [distance(group_manifold(A), mean(D0), mean(D)) for D in [D1, D_]]

    improvement_dB = -10*log10(last(dists) / first(dists))
    @test improvement_dB >= 18
end

# why is the error not exponential as in the localisation test?!?
