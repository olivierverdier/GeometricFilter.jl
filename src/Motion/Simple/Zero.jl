
"""
    ZeroMotion

Trivial rigid motion equal to zero everywhere.
"""
struct ZeroMotion{TA} <: AbstractRigidMotion{TA}
    A::TA
end

Base.show(io::IO, m::ZeroMotion) = print(io, "ZeroMotion($(m.A))")

rescale_motion(::Number, m::ZeroMotion) = m

function get_dynamics(m::ZeroMotion, ::Any)
    G = base_group(get_action(m))
    return zero_vector(G, identity_element(G))
end

function get_velocity(m::ZeroMotion)
    G = base_group(get_action(m))
    return zero_vector(G, identity_element(G))
end

function get_lin(m::ZeroMotion)
    G = base_group(get_action(m))
    return ξ -> get_velocity(m)
end

Base.:+(zm::ZeroMotion{TA}...)  where {TA} = first(zm)


Base.isapprox(M1::ZeroMotion{TA}, M2::ZeroMotion{TA}; kwargs...) where {TA} = true
