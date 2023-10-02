abstract type AbstractRigidMotion{TA} <: AbstractAffineMotion{TA} end

struct RigidMotion{TA,TV} <: AbstractRigidMotion{TA}
    A::TA # action G ⊂ Diff(M)
    vel::TV # in Alg(G)
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

