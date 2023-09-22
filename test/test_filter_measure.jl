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
    # D0 = ProjLogNormal(A, x0, cov, DefaultOrthonormalBasis())
    D0 = ProjLogNormal(A, x0, 1.)
    return D0
end



# G = MultiDisplacement(2)
G = SpecialOrthogonal(3)
A =  GroupOperationAction(G)
# A = DualGroupOperationAction(G)

M = group_manifold(A)

# G::MultiAffine, direction,
# OA = MultiAffineAction(G, direction)

function setup_action_observer(rng, OA,  N=10)
    refs = [rand(rng, group_manifold(OA)) for i in 1:N]
    obs = ProductObserver([ActionObserver(OA, ref) for ref in refs]...)
    return obs
end

# ground truth
D0 = setup_initial_dist(rng, A)

x0 = D0.μ
# only valid when M = G:
# x0 = identity_element(G)
# update_mean(D0, x0)



# obs = setup_action_observer(rng, MultiAffineAction(G, LeftAction()), 10)
# obs = setup_action_observer(rng, MultiAffineAction(G, RightAction()), 10)
V = Euclidean(3)
# obs = setup_action_observer(rng, RotationAction(V,G, LeftAction()), 10)
obs = setup_action_observer(rng, RotationAction(V,G, RightAction()), 10)
# obs = IdentityObserver(G)
m = obs(x0)

onoise = IsotropicNoise(get_manifold(obs), 1.)


run_filter(D0, N) = accumulate(1:N; init=D0) do D, i
    return update(D, Observation(obs, onoise, m))
end


# actually run the test
noise = ActionNoise(A, 1.0)
x1 = noise(rng, x0)
D1 = update_mean(D0, x1)

# Ds = _run(D1, 1000)
Ds = run_filter(D1, 3000)

gdist = [distance(M, D0.μ, D.μ) for D in Ds]

last(Ds).μ

# why is the error not exponential as in the localisation test?!?

