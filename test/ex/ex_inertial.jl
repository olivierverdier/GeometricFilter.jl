using GeometricFilter
using Manifolds
using MultiAffine

using PDMats
using LinearAlgebra

using SparseArrays

# Running example from https://github.com/CAOR-MINES-ParisTech/ukfm/blob/master/python/examples/inertial_navigation.py

include("inertial.jl")

const DIM = 3


#--------------------------------
# Constants
#--------------------------------

const SPAN = 30 # (s)
const FREQ = 100 # (Hz)
const RADIUS = 5

import Random

rng = Random.default_rng()

#--------------------------------
# Circular Trajectory
#--------------------------------

const τ = 2 * π

"""
    make_unit_circle

freq: frequence per turn (Float [Hz])

period: turn period (Float [s])
dim: total dimension (Integer ≥ 2)
turns: how many turns (Integer)
"""
function make_unit_circle(freq; period=1, dim=2, turns=1)
    n_pts = round(Int, period * freq * turns)
    span = period * turns
    ts = LinRange(0, span, n_pts)
    zs = exp.(im * ts / period * τ)
    xs = hcat(imag(zs), real(zs), zeros(length(zs), dim - 2))'
    return ts, xs
end






ts_, xs_ = make_unit_circle(FREQ; period=SPAN, dim=DIM)
xs__ = RADIUS * xs_
# to make sure initial velocity is zero:
# ts = vcat(-ts_[2], ts_)
# xs = [xs__[:,1] xs__]
ts = ts_
xs = xs__




#--------------------------------
# Observers
#--------------------------------


function make_landmark_observer(G, landmarks)
    obs_action = MultiAffineAction(G, [1.0, 0], RightAction())
    observers = [ActionObserver(obs_action, landmark) for landmark in eachcol(landmarks)]
    return ProductObserver(observers...)
end

G = GRAVITY[:G]

# Main observer
landmarks = hcat([0, 2, 2], [-2, -2, -2], [2, -2, -2])
observer = make_landmark_observer(G, landmarks)

# Convenience observers
pos_observer = PositionObserver(MultiAffineAction(G, [1, 0]))
vel_observer = PositionObserver(MultiAffineAction(G, [0, 1]))









#--------------------------------
# Singular Process Covariance
#--------------------------------

using PDMatsSingular

function make_diag_cov(G::MultiDisplacement, a_std, ω_std)
    D = spzeros(manifold_dimension(G))
    idx = first(axes(D))
    D[MultiAffine.normal_indices(G, idx; pos=1)] .= a_std^2
    D[MultiAffine.factor_indices(G, idx)] .= ω_std^2
    return Covariance(PDMats.PDiagMat(D))
end



#--------------------------------
# Process Noise
#--------------------------------

a_std = 0.01
ω_std = 0.01

proc_cov = make_diag_cov(G, a_std, ω_std)

# model process noise as right action (dual group action)
process_noise = ActionNoise(DualGroupOperationAction(G), proc_cov, DefaultOrthonormalBasis())


#--------------------------------
# Initial distribution
#--------------------------------

v = ones(3) / 3  * 10 * τ/360 * sqrt(2)
H = submanifold(G, 2)
ξ = get_vector_lie(H, v, DefaultOrthonormalBasis())
rot = exp_lie(H, ξ)


# diag_proc_cov = PDMats.PDiagMat(diag(proc_cov))


function make_initial_dist(pose, process_noise)
    pose_ = copy(pose)
    pose_.x[1][:, 1] += [1, 0.5, 0.7]
    pose_.x[2][:] = compose(H, pose.x[2], rot)
    # dist = ProjLogNormal(
    #     DualGroupOperationAction(G),
    #     pose_,
    #     # make_proc_cov(G, 1.0, 10 * τ / 360 * sqrt(2)),
    #     make_diag_cov(G, 1.0, 10 * τ / 360 * sqrt(2)),
    #     DefaultOrthonormalBasis(),)
    dist = ProjLogNormal(pose_,
        update_cov(process_noise, make_diag_cov(G, 1.0, 10 * τ / 360 * sqrt(2))))
    return dist
end


# diag_process_noise = ActionNoise(DualGroupOperationAction(G), diag_proc_cov, DefaultOrthonormalBasis())

# signal = generate_signal(rng, sms, pose)


"✓"
