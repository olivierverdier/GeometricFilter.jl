using Manifolds
using AffineMotions

using ManifoldGroupUtils
using RecursiveArrayTools


"""
    make_velocity(G::MultiAffineGroup, a, ω=nothing; pos=1)

Return the Lie Algebra element (0,a;ω)
"""
function make_velocity(G::MultiAffineGroup, a, ω=nothing; pos=1)
    normal = zero_vector(algebra(normal_group(G)))
    col_nb = first(Iterators.drop(axes(normal, 1), pos))
    normal[:,col_nb] = a
    if isnothing(ω)
        factor = zero_vector(algebra(factor_group(G)))
    else
        factor = get_vector_lie(factor_group(G), ω * sqrt(2), DefaultOrthonormalBasis())
    end
    return ArrayPartition(normal, factor)
end

"""
    get_inertial_motions(G, ::ActionDirection, ξ_body, ξ_nav)

Return a list of motions corresponding to the motion (with the standard left action of the group on itself):
```math
φ(χ) = ξ_{nav} + χ ξ_{body} χ^{-1} + ψ_M
```
where ``ψ_M`` is an adjoint linear motion corresponding to
the matrix ``M``.
"""
function get_inertial_motions(G, ::LeftAction, ξ_body, ξ_nav; M=[0 1;0 0])
    action = GroupOperationAction(G)
    m_nav = RigidMotion(action, ξ_nav)
    m_body = TranslationMotion(G, -ξ_body, LeftSide())
    m_ma = AdjointLinearMotion(G, M, LeftSide())
    return (m_body, m_nav, m_ma)
end

# -make_velocity(G, a...)

get_inertial_motions(G, ::RightAction, ξ_body, ξ_nav) = map(swap_group_motion, get_inertial_motions(G, LeftAction(), ξ_body, ξ_nav))

make_inertial_motion(G, AD::ActionDirection, ξ_body, ξ_nav) = sum(get_inertial_motions(G, AD, ξ_body, ξ_nav))

#--------------------------------
# Body accelerations from navigation accelerations
#--------------------------------

motion_from_sensors(grav, conv, sensors...) = let (G,g,Ξ) = grav
    make_inertial_motion(G, conv, make_velocity(G, sensors...), Ξ)
end

"""
    nav2body_accelerations(grav, pose, dts, as, ωs, g)

Convert navigation accelerations to body accelerations.
"""
function nav2body_accelerations(grav, pose, dtas, as, ωs)
    G = grav[:G]
    g = grav[:g]
    N = size(as, 2)
    poses = Vector{typeof(identity_element(G))}(undef, N + 1)
    body_accelerations = similar(as)
    H = submanifold(G, 2)
    rotacc = RotationAction(Euclidean(3), H)
    lastp = foldl(zip(axes(poses, 1), axes(body_accelerations, 2), dtas, eachcol(as), eachcol(ωs)); init=pose) do p, (pi, ai, dt, acc, ω)
        poses[pi] = p
        R = submanifold_component(p, 2)
        ωvec = get_vector_lie(H, sqrt(2) * ω, DefaultOrthonormalBasis())
        Rmot = RigidMotion(DualGroupOperationAction(H), -ωvec)
        R_ = integrate((dt/2) * Rmot, R)
        a = apply(rotacc, inv(H, R_), acc - g)
        # general formula:
        # act = MultiAffineAction(G, [0,1], LeftAction())
        # obs = PositionObserver(act)
        # a = apply(rotacc, inv(H, R), acc) - obs(adjoint_action(G, inv(G, p), ξ_nav))
        body_accelerations[:, ai] = a

        m = motion_from_sensors(grav, RightAction(), a, ω)
        p_ = integrate(dt * m, p)
        return p_
    end
    poses[lastindex(poses)] = lastp
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

_make_ωs(as, ::Nothing) = zero(as)
function _make_ωs(as, ωs::Vector)
    res = reduce(hcat, fill(ωs, size(as, 2)))
    return res
end
_make_ωs(::Any, ωs::Matrix) = ωs

motions_from_body_accelerations(grav, dtas, body_accelerations, ωs) =
    map(zip(dtas, eachcol(body_accelerations), eachcol(_make_ωs(body_accelerations, ωs)))) do (dt, a, ω)
        return dt * motion_from_sensors(grav, RightAction(), a, ω)
    end


function compute_body_accelerations(grav, ts, xs, ωs=nothing)
    tvs, vs = tdiff(ts, xs)
    tas, as = tdiff(tvs, vs)
    dtas = diff(tvs)
    pose = initial_pose(xs[:, 1], vs[:, 1])
    ωs_ = _make_ωs(as, ωs)
    poses, body_accelerations = nav2body_accelerations(grav, pose, dtas, as, ωs_)
    return poses, dtas, body_accelerations
end

"""
    motions_from_trajectory(grav::Tuple,
      ts, # Vector of length N
      xs, # dxN Matrix (d is the dimension)
         ωs # coordinates in the Lie algebra so(d)
      )

Compute inertial motions from a given trajectory given by
the vector `ts` and the matrix `xs`.
The angular velocity `ωs` may be either a vector
(in which case the rotation is supposed to be constant),
or a vector of the same size as `xs`.
`grav` is a tuple of the same form as the constant `GRAVITY`.
"""
function motions_from_trajectory(grav, ts, xs, ωs=nothing)
    poses, dtas, body_accelerations = compute_body_accelerations(grav, ts, xs, ωs)
    motions = motions_from_body_accelerations(grav, dtas, body_accelerations, ωs)
    return poses, motions
end

function make_gravity(g)
    dim = length(g)
    G = MultiDisplacementGroup(dim, 2)
    return (G=G, g=g, Ξ=make_velocity(G, g))
end

GRAVITY = make_gravity([0, 0, -9.82])

