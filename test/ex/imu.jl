
"""
Return the Lie Algebra element (0,a;0)
"""
function make_velocity(G, a; pos=2)
    ξ = zero_vector(G, Identity(G))
    submanifold_component(G, ξ, 1)[:, pos] = a
    return ξ
end

"""
Return the Lie Algebra element (0,a;ω)
"""
function make_velocity(G, a, ω)
    ξ = make_velocity(G, a)
    submanifold_component(G, ξ, 2)[:] = ω
    return ξ
end

"""
    get_imu_motions(G, ::ActionDirection, ξ_body, ξ_nav)

Return a list of motions corresponding to the motion
```math
φ(χ) = ξ_{nav} + χ ξ_{body} χ^{-1} + ψ_M
```
where ``ψ_M`` is a multi-affine motion corresponding to
the matrix ``M``.
"""
function get_imu_motions(G, ::LeftAction, ξ_body, ξ_nav; M=[0 0;1. 0])
    action = GroupOperationAction(G)
    m_nav = RigidMotion(action, ξ_nav)
    m_body = TranslationMotion(G, -ξ_body, LeftAction())
    m_ma = MultiAffineMotion(G, M, LeftAction())
    return (m_body, m_nav, m_ma)
end

# -make_velocity(G, a...)

get_imu_motions(G, ::RightAction, ξ_body, ξ_nav) = map(swap_group_motion, get_motions(G, LeftAction(), ξ_body, ξ_nav))

make_imu_motion(G, AD::ActionDirection, ξ_body, ξ_nav) = sum(get_imu_motions(G, AD, ξ_body, ξ_nav))
