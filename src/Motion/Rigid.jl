abstract type AbstractRigidMotion{TA} <: AbstractAffineMotion{TA} end
@doc raw"""
     RigidMotion(action::Action, ξ)

The rigid motion action of a group on a manifold.
The variable ``ξ`` is an element of the Lie algebra of the acting group `base_group(action)`.

The corresponding vector field ``f`` at a point ``x`` of the
manifold ``M`` (i.e., `M := group_manifold(action)`) is:
```math
f(x) = ξ⋅x
```
where ``⋅`` represents the infinitesimal action defined
by `action`.
"""
struct RigidMotion{TA,TV} <: AbstractRigidMotion{TA}
    A::TA # action G ⊂ Diff(M)
    vel::TV # in Alg(G)
    function RigidMotion(A, vel)
        G = base_group(A)
        Alg_G = TangentSpaceAtPoint(G, identity_element(G))
        if !is_point(Alg_G, vel)
            throw(ErrorException("Expected a point in Alg(G), but $vel ∉ Alg($G)"))
        end
        return new{typeof(A), typeof(vel)}(A, vel)
    end
end


RigidMotion(A) = ZeroMotion(A)

Base.show(io::IO, m::RigidMotion) = print(io, "RigidMotion($(m.A), $(m.vel))")

get_dynamics(m::RigidMotion, u) = copy(m.vel)

function get_lin(m::RigidMotion)
    A = get_action(m)
    G = base_group(A)
    vel = m.vel
    function lin(ξ)
        return lie_bracket(G, vel, ξ)
    end
    return lin
end

Base.:+(m1::RigidMotion{TA}, m2::RigidMotion{TA})  where {TA} = _add_rigid_motions(m1,m2)

function _add_rigid_motions(m1::RigidMotion{TA}, m2::RigidMotion{TA})  where {TA}
    action = get_action(m1)
    return RigidMotion(action, m1.vel + m2.vel)
end


"""
    ZeroMotion

Trivial rigid motion equal to zero everywhere.
"""
struct ZeroMotion{TA} <: AbstractRigidMotion{TA}
    A::TA
end

Base.show(io::IO, m::ZeroMotion) = print(io, "ZeroMotion($(m.A))")

function get_dynamics(m::ZeroMotion, ::Any)
    G = base_group(get_action(m))
    return zero_vector(G, identity_element(G))
end

function _vel(m::ZeroMotion)
    G = base_group(get_action(m))
    return zero_vector(G, identity_element(G))
end

function get_lin(m::ZeroMotion)
    G = base_group(get_action(m))
    return ξ -> _vel(m)
end

