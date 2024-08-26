#================================
  Largely Obsolete
================================#

@doc raw"""
    struct AffineMotion <: AbstractAffineMotion
        A # action G -> Diff(M)
        f # motion: M -> Alg(G)
        lin # linear part: operator Alg(G) ⊗ Alg(G)*
    end

General affine motion, defined from an action of a group ``G``
on a manifold ``M``,
a dynamics ``f \colon G \to \mathfrak{g}``,
 and the corresponding constant linear part `lin`,
a linear endormorphism of ``\mathfrak{g}``.
"""
struct AffineMotion{TA<:AbstractGroupAction{LeftAction},TF,TL} <: AbstractAffineMotion{TA}
    A::TA # action G -> Diff(M)
    f::TF # motion: M -> Alg(G)
    lin::TL # linear part: operator Alg(G) ⊗ Alg(G)*
end

Base.show(io::IO, m::AffineMotion) = print(io, "AffineMotion($(m.A), [...])")

get_dynamics(m::AffineMotion, x) = m.f(x)

@doc raw"""
    get_lin(m::AffineMotion) :: Function

The linear part of the motion, a linear endomorphism of ``\mathfrak{G}``.
"""
get_lin(m::AffineMotion) = m.lin


# Base.:+(m1::AbstractAffineMotion{TA}, m2::AbstractAffineMotion{TA})  where {TA} = _add_affine_motions(m1,m2)

function _add_affine_motions(m1::AbstractAffineMotion{TA}, m2::AbstractAffineMotion{TA})  where {TA}
    # assert_equal_actions(m1, m2, "Cannot add motion with different actions")
    a1 = get_action(m1)
    function f(u)
        return get_dynamics(m1, u) + get_dynamics(m2, u)
    end
    l1 = get_lin(m1)
    l2 = get_lin(m2)
    function lin(ξ)
        return l1(ξ) + l2(ξ)
    end
    return AffineMotion(a1, f, lin)
end


# TODO: is it ever useful? what about the dual version?
function lift_at(m::AbstractAffineMotion, x)
    A = get_action(m)
    return AffineMotion(
        GroupOperationAction(base_group(A)),
        χ -> m(apply(A, χ, x)),
        get_lin(m)
    )
end

# Fallback for rescaling
function rescale_motion(s::Number, m::AbstractAffineMotion)
    A = get_action(m)
    lin = get_lin(m)
    return AffineMotion(A, x -> s*m(x), ξ -> s*lin(ξ))
end

function _swap_group_motion(m::AbstractAffineMotion)
    A = get_action(m)
    G = base_group(A)
    lin = get_lin(m)
    # new_f(χ) = -adjoint_action(G, _swap_inv(A, G, χ), m(χ))
    new_f(χ) = -_swap_adjoint_action(A, G, χ, m(χ))
    φ1 = m(identity_element(G))
    new_lin(ξ) = lin(ξ) - lie_bracket(G, φ1, ξ)
    return AffineMotion(_swap_group_action(A), new_f, new_lin)
end

swap_group_motion(m::AffineMotion{TA}) where {TA<:GroupOperationAction} = _swap_group_motion(m)
