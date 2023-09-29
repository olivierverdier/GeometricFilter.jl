"""
    MultiAffineAction(
      group::MultiAffine,
      selector::AbstractVector,
      conv::ActionDirection=LeftAction()
      )

Given a fixed vector ``S`` of size ``k`` (the `selector`),
this defines an action of the element ``[X;R]`` of the [`MultiAffine`](@ref) group (so ``X`` is a ``n×k`` matrix and ``R`` is an element of a matrix group)
 on the vector ``p`` of size ``n``.
The action is defined by ``[X;R]⋅p := XS+Rp``.
"""
struct MultiAffineAction{TH,dim,size,𝔽,TAD<:ActionDirection,TS<:AbstractVector} <: AbstractGroupAction{TAD}
    group::MultiAffine{TH,dim,size,𝔽}
    selector::TS # vector of length `size`
end

Base.show(io::IO, A::MultiAffineAction{<:Any,<:Any,<:Any,<:Any,TAD}) where {TAD} = print(io, "MultiAffineAction($(A.group), $(A.selector), $TAD())")

function MultiAffineAction(
    group::MultiAffine{TH, dim, size, 𝔽},
    selector,
    conv::ActionDirection=LeftAction()
    ) where {TH, dim, size, 𝔽}
    @assert Base.size(selector, 1) == size
    return MultiAffineAction{TH, dim, size, 𝔽, typeof(conv), typeof(selector)}(group, selector)
end

"""
In the case k=1, this is the standard affine action.
"""
function MultiAffineAction(
    group::MultiAffine{TH, dim, 1, 𝔽},
    conv::ActionDirection=LeftAction()
    ) where {TH, dim, 𝔽}
    return MultiAffineAction(group, [1], conv)
end

function Manifolds.switch_direction(A::MultiAffineAction{TH,dim,size,𝔽,TAD}) where {TH,dim,size,𝔽,TAD}
    return MultiAffineAction(A.group, A.selector, switch_direction(TAD()))
end


Manifolds.base_group(A::MultiAffineAction) = A.group
Manifolds.group_manifold(::MultiAffineAction{G,dim,size,𝔽}) where {G,dim,size,𝔽} = Euclidean(dim; field=𝔽)

get_selector(A::MultiAffineAction) = A.selector

Manifolds.apply(::MultiAffineAction{TH,dim,size,𝔽,conv},
                ::Identity{MultiAffineOp{TH,dim,size,𝔽}}, p) where {TH,dim,size,𝔽,conv} = p

function Manifolds.apply!(
    A::MultiAffineAction{TH,dim,size,𝔽,LeftAction},
    q,
    χ,
    p, # vector of size dim
    ) where {TH,dim,size,𝔽}
    G = base_group(A)
    M,R = submanifold_components(G, χ)
    sel = get_selector(A)
    # compute: M*sel + R*p
    LinearAlgebra.mul!(q, M, sel)
    LinearAlgebra.mul!(q, R, p, 1, 1)
    return q
end


Manifolds.apply(A::MultiAffineAction{TH,dim,size,𝔽,RightAction}, a, p) where {TH,dim,size,𝔽} = apply(switch_direction(A), inv(base_group(A), a), p)



function Manifolds.apply_diff_group(
    A::MultiAffineAction{TH,dim,size,𝔽,LeftAction},
    ::Identity{MultiAffineOp{TH,dim,size,𝔽}},
    ξ,
    p
    ) where {TH,dim,size,𝔽}
    G = base_group(A)
    μ, ω = submanifold_components(G, ξ)
    sel = get_selector(A)
    @assert is_point(group_manifold(A), p)
    return μ*sel + ω*p
end


function Manifolds.apply_diff_group(
    A::MultiAffineAction{TH,dim,size,𝔽,RightAction},
    I::Identity{MultiAffineOp{TH,dim,size,𝔽}},
    ξ,
    p
    ) where {TH,dim,size,𝔽}
    return apply_diff_group(switch_direction(A), I, -ξ, p)
end
