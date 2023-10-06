using Manifolds

"""
Return the Lie Algebra element (0,a;0)
"""
function make_velocity_(G, a; pos=1)
    ξ = zero_vector(G, Identity(G))
    X = submanifold_component(G, ξ, 1)
    idx = first(Iterators.drop(axes(X, 2), pos))
    X[:, idx] = a
    return ξ
end

"""
Return the Lie Algebra element (0,a;ω)
"""
function make_velocity_(G, a, ω; pos=1)
    ξ = make_velocity_(G, a; pos)
    submanifold_component(G, ξ, 2)[:] = ω
    return ξ
end

function make_velocity(G, a, ω=nothing; pos=1)
    vector = zeros(manifold_dimension(G))
    idx = first(axes(vector))
    a_idx = GeometricFilter.normal_indices(G, idx; pos=pos)
    ω_idx = GeometricFilter.factor_indices(G, idx)
    vector[a_idx] .= a
    if ω !== nothing
      vector[ω_idx] .= ω
    end
    return get_vector_lie(G, vector, DefaultOrthonormalBasis())
end

"""
    get_inertial_motions(G, ::ActionDirection, ξ_body, ξ_nav)

Return a list of motions corresponding to the motion (with the standard left action of the group on itself):
```math
φ(χ) = ξ_{nav} + χ ξ_{body} χ^{-1} + ψ_M
```
where ``ψ_M`` is a multi-affine motion corresponding to
the matrix ``M``.
"""
function get_inertial_motions(G, ::LeftAction, ξ_body, ξ_nav; M=[0 0;1. 0])
    action = GroupOperationAction(G)
    m_nav = RigidMotion(action, ξ_nav)
    m_body = TranslationMotion(G, -ξ_body, LeftAction())
    m_ma = MultiAffineMotion(G, M, LeftAction())
    return (m_body, m_nav, m_ma)
end

# -make_velocity(G, a...)

get_inertial_motions(G, ::RightAction, ξ_body, ξ_nav) = map(swap_group_motion, get_inertial_motions(G, LeftAction(), ξ_body, ξ_nav))

make_inertial_motion(G, AD::ActionDirection, ξ_body, ξ_nav) = sum(get_inertial_motions(G, AD, ξ_body, ξ_nav))

#--------------------------------
# Body accelerations from navigation accelerations
#--------------------------------

SIZE = 2

g₀ = [0, 0, -9.82]
G = MultiDisplacement(3,2)
Ξ_nav = make_velocity(G, g₀)

motion_from_sensors(G, conv, sensors...) = make_inertial_motion(G, conv, make_velocity(G, sensors...), Ξ_nav)


function nav2body_accelerations(G, dtas, as, pose; g_vec=g₀)
    acc_act = MultiAffineAction(G, [0,1], RightAction())

    sim_res = accumulate(zip(dtas, eachcol(as)); init=(pose, nothing)) do (p_,_a), (dt,acc)
        p = copy(p_)
        p.x[1][:,2] = g_vec
        a = apply(acc_act, p, acc)
        m = motion_from_sensors(G, RightAction(), a)
        p__ = integrate(dt * m, p_)
        return (p__, a)
    end
    poses = first.(sim_res)
    body_accelerations = last.(sim_res)
    return poses, body_accelerations
end

#--------------------------------
# Initial Pose
#--------------------------------

function initial_pose(x,v)
    pose = identity_element(G)
    pose.x[1][:, 1] = x
    pose.x[1][:, 2] = v
    return pose
end


#--------------------------------
# Motion from given trajectory
#--------------------------------
function tdiff(ts, xs)
    dt = diff(ts)
    tm = map(zip(ts, Iterators.drop(ts, 1))) do (t0, t1)
        return (t0 + t1) / 2
    end
    dx = diff(xs; dims=2)
    return tm, dx ./ dt'
end

motions_from_body_accelerations(G, dtas, body_accelerations) = map(zip(dtas, body_accelerations)) do (dt, a)
    return dt * motion_from_sensors(G, RightAction(), a)
end


# motions = compute_motions(G, dtas, body_accelerations)

function compute_body_accelerations(G, ts, xs; kwargs...)
    tvs, vs = tdiff(ts, xs)
    tas, as = tdiff(tvs, vs)
    dtas = diff(tvs)
    pose = initial_pose(xs[:, 1], vs[:, 1])
    poses, body_accelerations = nav2body_accelerations(G, dtas, as, pose; kwargs...)
    return prepend!(poses, [pose]), dtas, body_accelerations
end

function motions_from_trajectory(G, ts, xs; kwargs...)
    poses, dtas, body_accelerations = compute_body_accelerations(G, ts, xs; kwargs...)
    motions = motions_from_body_accelerations(G, dtas, body_accelerations)
    return poses, motions
end
