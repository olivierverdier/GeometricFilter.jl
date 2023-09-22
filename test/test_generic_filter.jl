using Test
using GeometricFilter
using Manifolds


import PDMats

import Random

rng = Random.default_rng()

function setup_static_motion(G, dev=0)
    # ξ = rand(rng, G; vector_at=Identity(G))
    ξ = zero_vector(G, Identity(G))
    motion = RigidMotion(A, ξ)
    # zero process noise instead?
    pnoise = IsotropicNoise(G, x->dev)
    return motion, pnoise
end

function setup_id_observer(A, dev=1.)
    M = group_manifold(A)
    obs = IdentityObserver(M) # or specified?
    onoise = IsotropicNoise(get_manifold(obs), x->dev)
    return obs, onoise
end

function setup_initial_dist(rng, A, dev=1.)
    x0 = rand(rng, group_manifold(A))
    # cov = dev^2*PDMats.PDiagMat(ones(manifold_dimension(base_group(A))))
    cov = PDMats.ScalMat(manifold_dimension(base_group(A)), dev^2)
    D0 = ProjLogNormal(A, x0, cov, DefaultOrthonormalBasis())
    return D0
end



# G = MultiDisplacement(2,1)
# A = GroupOperationAction(G)
# M = group_manifold(A)

G = MultiDisplacement(2)
A = DualGroupOperationAction(G)

# M = Euclidean(2)

M = group_manifold(A)

motion, pnoise = setup_static_motion(base_group(A), 1.)
# obs, onoise = setup_id_observer(A)

OA = MultiAffineAction(G, RightAction())
# N = get_manifold(OA)
ref = ones(manifold_dimension(group_manifold(OA)))
obs = ActionObserver(OA, ref)

onoise = IsotropicNoise(get_manifold(obs), x -> 1.)
D0 = setup_initial_dist(rng, A)

# only valid when M = G:
x0 = identity_element(G)
update_mean(D0, x0)

function noisy_motion(A::DualGroupOperationAction, rng, x1) 
    G = base_group(A)
    x1_ = pnoise(rng, x1)
    # only valid if M == G and DualGroupAction
    vel_ = log_lie(G, compose(G, inv(G, D0.μ), x1_))
    motion_ = RigidMotion(A, vel_)
    return motion_
end


function run(rng, D0)
    x1 = integrate(motion, D0.μ)
    y1 = onoise(rng, obs(x1))

    motion_ = noisy_motion(A, rng, x1)

    D1 = predict(D0, motion_, pnoise)
    D1_ = update(D1, obs, onoise, y1)

    return D1,D1_
end

D1,D1_ = run(rng, D0)

#     # m_dist = distance(get_manifold(obs), obs(D0.μ), y1)
i_dist = distance(M, D1.μ, D0.μ)
c_dist = distance(M, D0.μ, D1_.μ)
#     return [i_dist, c_dist]
# end

# distances = hcat([get_distances(rng) for i in 1:100]...)
# mean(distances[2,:] - distances[1,:])


@show i_dist
@show c_dist
# @assert c_dist <= m_dist

