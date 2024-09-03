using Test
using GeometricFilter
using AffineMotions
using ManifoldNormal
using MultiAffine
using Manifolds
using RecursiveArrayTools

τ = 2 * π
G = MultiDisplacement(2)

#--------------------------------
# Create path
#--------------------------------

straight_path(dir, dt, scale) = dir * LinRange(0, scale, Int(scale / dt) + 1)
make_rotation(θ) = exp(θ * [0 1; -1 0]')
pose_mat(z, θ) = ArrayPartition([real(z); imag(z);;], make_rotation(θ))

# scale = 30
# scale = 5
# scale = 1
# dt = 0.2

"""
    make_square_path(scale, dt)

A square trajectory in the complex plane,
each side has length `scale`,
and is discretized every `dt` points.
For instance, `make_square(2.0, 0.1)` creates
a square of size 2 with 20 points per side.
"""
function make_square_path(scale; dt=0.2)
    dirs = Complex{Float64}[exp(im * j * τ / 4) for j in 0:3]
    paths = [straight_path(d, dt, scale) for d in dirs]
    translated_paths = accumulate(paths; init=(LinRange(0,0,0), 0im)) do (_p,s),p
        p_ = p.+s
        return (p_, last(p_))
    end
    full_path = reduce(vcat, first.(translated_paths))
    return full_path
end

make_square_path(scale, turns; kwargs...) = reduce(vcat, fill(make_square_path(scale; kwargs...), turns))


function compute_poses(full_path)
    angles = [imag(log(full_path[i+1] - full_path[i])) for i in 1:length(full_path)-1]
    return [pose_mat(z, angle) for (z, angle) in zip(full_path, angles)]
end



@doc raw"""
    compute_velocities(G, poses)

Compute ``ξ_i`` such that ``χ_{i+1} = χ_i \exp(-ξ_i)``
 so, ``ξ_i = -\log(g_i^{-1} g_{i+1})``.
"""
compute_velocities(G, poses) = [-log_lie(G, compose(G, inv(G, poses[i]), poses[i+1])) for i in 1:length(poses)-1]


# TODO: comment on the following choice:

A = DualGroupOperationAction(G)
compute_motions(A, vels) = map(vel -> RigidMotion(A, vel), vels)

# A = GroupOperationAction(G)
# motions = [TranslationMotion(A, vel) for vel in vels]

#--------------------------------
# Observers from Landmarks
#--------------------------------

# Normalised landmarks, located in a square of length one.
LANDMARKS = hcat([
    [0.2, 0.2],
    [0.5, 0.3],
    [0.8, 0.2],
    [0.7, 0.5],
    [0.8, 0.8],
    [0.5, 0.7],
    [0.2, 0.8],
    [0.3, 0.5]]...)

import NearestNeighbors: KDTree, knn

make_landmarks(scale, landmarks=LANDMARKS) = scale*landmarks
make_tree(scale, landmarks=LANDMARKS) = KDTree(make_landmarks(scale, landmarks))

"""
    landmark_observer(G :: DecoratedManifold,
     landmark :: AbstractVector) :: AbstractObserver

Distance and bearing observer corresponding to one landmark.
"""
landmark_observer(G::MultiAffineGroup, landmark) = ActionObserver(MultiAffineAction(G, RightAction()), landmark)
# make_observer(G::MultiAffineGroup, landmark) = ActionObserver(MultiAffineAction(G, LeftAction()), landmark)

"""
    full_observer(G, pose, tree, k=2) :: AbstractObserver

Create a full observer given a `pose`, a tree and the number `k` of closest landmarks to use.
"""
function full_observer(G, pose, tree; k=2)
    refs = tree.data[first(knn(tree, vec(submanifold_component(pose, 1)), k))]
    observers = [landmark_observer(G, ref) for ref in refs]
    observer = ProductObserver(observers...)
    return observer
end


#--------------------------------
# Prepare Simulation
#--------------------------------


# cposes = generate_signal(motions, first(poses))

import PDMats


process_action = A
# process_action = GroupOperationAction(G)
# process_noise = ActionNoise(process_action, PDMats.PDiagMat([.01,.01,.015].^2), DefaultOrthogonalBasis())
process_noise = ActionNoise(DualGroupOperationAction(G), PDMats.PDiagMat([.01,.01,sqrt(2)*.015].^2), DefaultOrthonormalBasis())

dist = ProjLogNormal(identity_element(G), update_cov(process_noise, 1.))

let
    using Random
    rng = Random.default_rng()
    FREQ = 5
    # full_path = make_square_path(30, 1; dt=1/FREQ)
    # poses = compute_poses(full_path)
    poses = fill(identity_element(G), 3)
    vels = compute_velocities(G, poses)
    motions = compute_motions(A, vels)
    # just checking that it works in principle:
    simulate_filter(dist, [StochasticMotion(first(motions), process_noise)], [Observation()], SensorPerturbation(rng))
    " "
end


# make_obs_noise(M) = IsotropicNoise(M, 0.5)


