
struct MultiAffineMotion{TG,TM,TAD<:ActionDirection} <: AbstractAffineMotion
    G::TG # MultiAffine{H, dim, size, 𝔽}
    M::TM # Array{𝔽, 2}; size×size array
end

@doc raw"""
    MultiAffineMotion(G::DecoratedManifold,M::Array{𝔽, 2},conv::ActionDirection)

One of the affine motions which are neither rigid motions nor translations on the group SE_k(n) (or more general affine groups). If a group element is denoted by \(χ = [X;R]\), then the motion corresponding to the \(k×k\) matrix \(M\) is defined by \(φ(χ) := (XM;0)\).
"""
MultiAffineMotion(G,M,conv) = MultiAffineMotion{typeof(G), typeof(M), typeof(conv)}(G,M)

get_action(m::MultiAffineMotion{TG,TM,LeftAction}) where {TG,TM} = GroupOperationAction(m.G)
get_action(m::MultiAffineMotion{TG,TM,RightAction}) where {TG,TM} = DualGroupOperationAction(m.G)

function _lin(m, χ)
    X = submanifold_component(m.G, χ, 1)
    res = X*m.M
    return from_normal_alg(m.G, res)
end

get_dynamics(m::MultiAffineMotion{TG,TM,LeftAction}, χ) where {TG,TM} = _lin(m, χ)
function get_dynamics(m::MultiAffineMotion{TG,TM,RightAction}, χ) where {TG,TM}
    R = to_factor_grp(m.G, χ)
    tmp = _lin(m, χ)
    tmp_ = submanifold_component(m.G, tmp, 1)
    A = m.G.op.action
    H = submanifold(m.G,2)
    tmp_[:] = -apply(A, inv(H, R), tmp_)
    return tmp
end

get_lin(m::MultiAffineMotion) = ξ -> _lin(m, ξ)

swap_group_motion(m::MultiAffineMotion{TG,TM,TAD}) where {TG,TM,TAD} = MultiAffineMotion(m.G, m.M, switch_direction(TAD()))
