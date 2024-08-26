abstract type AbstractRigidMotion{TA} <: SimpleAffineMotion{TA} end

#--------------------------------
# AbstractRigidMotion Interface
#--------------------------------
@doc raw"""
     get_velocity(m::AbstractRigidMotion)

Access to the underlying infinitesimal motion defining the rigid motion `m`.
"""
function get_velocity end
#--------------------------------



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
        Alg_G = TangentSpace(G, identity_element(G))
        if !is_point(Alg_G, vel)
            throw(ErrorException("Expected a point in Alg(G), but $vel ∉ Alg($G)"))
        end
        return new{typeof(A), typeof(vel)}(A, vel)
    end
end

Base.isapprox(M1::AbstractRigidMotion, M2::AbstractRigidMotion; kwargs...) = isapprox(get_velocity(M1), get_velocity(M2); kwargs...)

RigidMotion(A) = ZeroMotion(A)

Base.show(io::IO, m::RigidMotion) = print(io, "RigidMotion($(m.A), $(m.vel))")

rescale_motion(s::Number, m::RigidMotion) = RigidMotion(m.A, s*m.vel)

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

get_velocity(m::RigidMotion) = m.vel

Base.:+(rm::AbstractRigidMotion{TA}...)  where {TA} = _add_rigid_motions(rm...)

function _add_rigid_motions(rms::AbstractRigidMotion{TA}...)  where {TA}
    action = get_action(first(rms))
    return RigidMotion(action, sum([get_velocity(rm) for rm in rms]))
end

Base.isapprox(M1::RigidMotion{TA}, M2::RigidMotion{TA}; kwargs...) where {TA} = isapprox(M1.vel, M2.vel; kwargs...)

include("Zero.jl")
