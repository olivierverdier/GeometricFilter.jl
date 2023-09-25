
struct RigidMotion{TA,TV} <: AbstractAffineMotion{TA}
    A::TA # action G ⊂ Diff(M)
    vel::TV # in Alg(G)
end

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
