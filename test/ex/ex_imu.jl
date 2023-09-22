using GeometricFilter
using Manifolds


import Random

rng = Random.default_rng()

τ = 2*π

function make_circle(span, freq, radius, dim)
    n_pts = span * freq
    dt = 1/freq # (s)
    ts = LinRange(0, span, n_pts)
    zs = radius*exp.(im*ts*τ)
    xs = hcat(imag(zs),real(zs),zeros(length(zs), dim-2))'
    return ts, xs
end

function tdiff(ts, xs)
    dt = diff(ts)
    tm = zeros(size(dt))
    for (i, t0, t1) in zip(eachindex(tm), ts, Iterators.drop(ts,1))
        m = (t0+t1)/2
        tm[i] = m
    end
    dx = diff(xs; dims=2)
    return tm, dx./dt'
end



# function zdiff(xs::AbstractArray{T,2}) where {T}
#     vs = similar(xs)
#     vs[:,1] .= 0
#     vs[:,2:end] = diff(xs; dims=2)
#     return vs
# end

# zdiff(ts::AbstractArray{T,1}) where {T} = zdiff(reshape(ts, 1, :))


# ddts, ddxs = tdiff(dts, dxs)


# vels = zdiff(xs)./zdiff(ts)
# accs = zdiff(vels)./zdiff(ts)

# submanifold_component(G, pose, 1)[:] = hcat(xs[:,1], vels[:,1])
# fill!(submanifold_component(G, pose, 1), 0)


include("imu.jl")


DIM = 3
SIZE = 2


G = MultiDisplacement(DIM, SIZE)

g_vec = [0, 0, -9.82]
Ξ_nav = make_velocity(G, g_vec)

SPAN = 30 # (s)
FREQ = 100 # (Hz)
RADIUS = 5
ts, xs = make_circle(SPAN, FREQ, RADIUS, DIM)

tvs, vs = tdiff(ts, xs)
tas, as = tdiff(tvs, vs)

# tas_, dtas = tdiff(tvs, reshape(tvs, 1, :))
dtas = diff(tvs)


pose = identity_element(G)
pose.x[1][:,1] = xs[:,1]
pose.x[1][:, 2] = vs[:, 1]



motion_from_sensors(G, sens...) = make_imu_motion(G, LeftAction(), make_velocity(G, sens...), Ξ_nav)


function simulate(G, dtas, as, pose)
    poses = [pose]
    accelerations = []
    acc_act = MultiAffineAction(G, [0,1], RightAction())

    for (dt, acc) in zip(dtas, eachcol(as))
        p = copy(last(poses))
        p.x[1][:,2] = g_vec
        a = apply(acc_act, p, acc)
        push!(accelerations, a)
        m = motion_from_sensors(G, a)
        p_ = integrate(dt * m, p)
        push!(poses, p_)
    end
    return poses, accelerations
end

pos_action = MultiAffineAction(G, [1,0])
pos_observation = PositionObserver(pos_action)
vel_action = MultiAffineAction(G, [0, 1])
vel_observation = PositionObserver(vel_action)

landmarks = [[0, 2, 2], [-2, -2, -2], [2, -2, -2]]

obs_action = MultiAffineAction(G, [1.,0], RightAction())
observer = ProductObserver([ActionObserver(obs_action, landmark) for landmark in landmarks]...)

# simulate noisy observation from true trajectory
# simulate noisy motions? or simulate noise on poses?


poses, accelerations = simulate(G, dtas, as, pose)

a_std = 0.01
ω_std = 0.01
function make_proc_cov_mat(G::MultiDisplacement{dim,size}, a_std, ω_std) where {dim, size}
    std_vec = zeros(manifold_dimension(G))
    idx = axes(std_vec, 1)
    a_idx = collect(Iterators.take(Iterators.drop(idx, dim), dim))
    ω_idx = collect(Iterators.drop(idx, size * dim))
    std_vec[a_idx] .= a_std
    std_vec[ω_idx] .= ω_std
    proc_cov = PDMats.PDiagMat(std_vec .^ 2)
    return proc_cov
end

function make_injection(dof, idx)
    mat = zeros(dof, length(idx))
    for (i,j) in zip(idx, axes(mat, 2))
        mat[i,j] = 1.
    end
    return mat
end


"""Indices for acceleration"""
a_idx(dim, size, idx) = collect(Iterators.take(Iterators.drop(idx, dim), dim))
"""Indices for angular velocity"""
ω_idx(dim, size, idx) = collect(Iterators.drop(idx, size * dim))

make_singular_cov(dof, dim, size, f) = make_injection(dof, f(dim, size, Base.OneTo(dof)))

function make_proc_cov(G::MultiDisplacement{dim,size}, a_std, ω_std) where {dim, size}
    dof = manifold_dimension(G)
    a_cov = Covariance(make_singular_cov(dof, dim, size, a_idx))
    ω_cov = Covariance(make_singular_cov(dof, dim, size, a_idx))
    return a_cov + ω_cov
end

proc_cov = make_proc_cov(G, a_std, ω_std)

# model process noise as right action (dual group action)
process_noise = ActionNoise(DualGroupOperationAction(G), proc_cov, DefaultOrthonormalBasis())

obs_std = 1.0
obs_noise = IsotropicNoise(get_manifold(observer), obs_std)



dist = ProjLogNormal(
    DualGroupOperationAction(G),
    pose,
    make_proc_cov(G, 1., 10*τ/360),
    DefaultOrthonormalBasis(),)


sim_poses = [pose]
for (dt, a) in zip(dtas, eachcol(as))
    println(dt)
    d = last(dists)
    predict(d, dt * motion_from_sensors(G, a), process_noise)
    # d_ = process_noise(rng, d)
    push!(dists, d_)
end

dists = [dist]
for (dt, a) in zip(dtas, eachcol(as))
    println(dt)
    d = last(dists)
    predict(d, dt*motion_from_sensors(G, a), process_noise)
    # d_ = process_noise(rng, d)
    push!(dists, d_)
end
# observation frequency: one every second

"✓"
