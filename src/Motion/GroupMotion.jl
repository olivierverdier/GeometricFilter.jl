
"""
    swap_group_motion(::Motion)

Swap from standard to dual group action, while preserving the same dynamics.
"""
swap_group_motion(m::RigidMotion{TA}) where {TA<:GroupOperationAction} = TranslationMotion(base_group(get_action(m)), m.vel, RightAction())
swap_group_motion(m::TranslationMotion{RightAction}) = RigidMotion(GroupOperationAction(m.G), m.vel)
swap_group_motion(m::RigidMotion{TA}) where {TA<:DualGroupOperationAction} = TranslationMotion(base_group(get_action(m)), m.vel, LeftAction())
swap_group_motion(m::TranslationMotion{LeftAction}) = RigidMotion(DualGroupOperationAction(m.G), m.vel)

# _swap_inv(::GroupOperationAction, G, χ) = inv(G,χ)
# _swap_inv(::DualGroupOperationAction, G, χ) = χ
_swap_adjoint_action(::GroupOperationAction, G, χ, ξ) = inverse_adjoint_action(G, χ, ξ)
_swap_adjoint_action(::DualGroupOperationAction, G, χ, ξ) = adjoint_action(G, χ, ξ)
_swap_group_action(A::GroupOperationAction) = DualGroupOperationAction(base_group(A))
_swap_group_action(A::DualGroupOperationAction) = GroupOperationAction(base_group(A))

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

swap_group_motion(m::AffineMotion{TA}) where {TA<:Union{GroupOperationAction,DualGroupOperationAction}} = _swap_group_motion(m)
