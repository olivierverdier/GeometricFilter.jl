
struct AdjointLinearMotion{TAD<:Manifolds.GroupActionSide, TA, TG,TM} <: AffineMotions.SimpleAffineMotion{TA}
    G::TG # MultiAffine{H, dim, size, 𝔽}
    M::TM # Array{𝔽, 2}; size×size array
end

Base.show(io::IO, m::AdjointLinearMotion{TAD}) where {TAD} = print(io, "AdjointLinearMotion($(m.G), $(m.M), $TAD())")

rescale_motion(s::Number, m::AdjointLinearMotion{TAD}) where {TAD} = AdjointLinearMotion(m.G, s*m.M, TAD())

@doc raw"""
    AdjointLinearMotion(G::DecoratedManifold,M::Array{𝔽, 2},conv::ActionDirection)

One of the affine motions which are neither rigid motions nor translations on the group ``SE_k(n)`` (or more general affine groups). If a group element is denoted by ``χ = [X;R]``, then the motion corresponding to the ``k×k`` matrix ``M`` is defined by ``φ(χ) := (XM^*;0)``.
"""
AdjointLinearMotion(G, M, conv) = AdjointLinearMotion{typeof(conv),typeof(_get_group_operation_action(G, conv)),typeof(G),typeof(M)}(G, M)

get_action(m::AdjointLinearMotion{TAD}) where {TAD} = _get_group_operation_action(m.G, TAD())

function _lin(m, χ)
    X = submanifold_component(m.G, χ, 1)
    res = X*m.M'
    return from_normal_alg(m.G, res)
end

get_dynamics(m::AdjointLinearMotion{LeftSide}, χ)  = _lin(m, χ)
function get_dynamics(m::AdjointLinearMotion{RightSide}, χ) 
    R = to_factor_grp(m.G, χ)
    tmp = _lin(m, χ)
    tmp_ = submanifold_component(m.G, tmp, 1)
    A = m.G.op.action
    H = submanifold(m.G,2)
    tmp_[:] = -apply(A, inv(H, R), tmp_)
    return tmp
end

get_lin(m::AdjointLinearMotion) = ξ -> _lin(m, ξ)

swap_group_motion(m::AdjointLinearMotion{TAD}) where {TAD} = AdjointLinearMotion(m.G, m.M, switch_side(TAD()))

Base.:+(m1::AdjointLinearMotion{<:Any, TA}, m2::AdjointLinearMotion{<:Any, TA})  where {TA} = _add_multiaffine_motions(m1,m2)

function _add_multiaffine_motions(m1::AdjointLinearMotion{TAD}, m2::AdjointLinearMotion{TAD}) where {TAD}
    return AdjointLinearMotion(m1.G, m1.M+m2.M, TAD())
end

Base.isapprox(M1::AdjointLinearMotion{<:Any, TA}, M2::AdjointLinearMotion{<:Any,TA}; kwargs...) where {TA} = isapprox(M1.M, M2.M; kwargs...)
