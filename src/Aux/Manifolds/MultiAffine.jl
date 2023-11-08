



const MultiColumnwiseMultiplicationAction{G,dim,size,𝔽} = Manifolds.ColumnwiseMultiplicationAction{
    LeftAction,
    TranslationGroup{ManifoldsBase.TypeParameter{Tuple{dim,size}},𝔽},
    G,
}

const MultiAffine{G,dim,size,𝔽} = SemidirectProductGroup{
    𝔽,
    TranslationGroup{ManifoldsBase.TypeParameter{Tuple{dim,size}},𝔽},
    G,
    MultiColumnwiseMultiplicationAction{G,dim,size,𝔽}
}

const MultiAffineOp{G,dim,size,𝔽} = Manifolds.SemidirectProductOperation{MultiColumnwiseMultiplicationAction{G,dim,size,𝔽}}

@doc raw"""
    MultiAffine(G, k=1)

An affine group modelling matrices of the form
```math
χ = \begin{bmatrix}
\mathbf{1} & \mathbf{0} \\
X & g
```
where ``g`` is a matrix element of the group ``G``, represented in dimension ``n``,
and ``X`` is a ``n × k`` matrix.
If we denote such an element by ``[X,g]``,
the multiplication law is ``[X,g] [X',g'] = [X+gX';gg']``.
"""
function MultiAffine(G::Manifolds.GeneralUnitaryMultiplicationGroup{ManifoldsBase.TypeParameter{Tuple{dim}},𝔽}, size::Integer=1) where {dim, 𝔽}
    space = TranslationGroup(dim,size;field=𝔽)
    action = Manifolds.ColumnwiseMultiplicationAction(space, G)
    group = GroupManifold(ProductManifold(space, G), Manifolds.SemidirectProductOperation(action))
    return group
end


Base.show(io::IO, ::MultiAffine{G, <:Any,size}) where {G, size} = print(io, "MultiAffine($(G), $(size))")

_get_representation_dim(::MultiAffine{<:Any,dim,size}
                        ) where {dim,size} = dim+size


function Manifolds.allocate_result(G::MultiAffine, ::Union{typeof(affine_matrix),typeof(screw_matrix)}, Xis...)
    d = _get_representation_dim(G)
    return allocate(Xis[1], Manifolds.Size(d,d))
end

Base.@propagate_inbounds function Manifolds._padvector!(
    ::MultiAffine{<:Any, <:Any, size},
    X::AbstractMatrix,
) where {size}
    for i in Base.Iterators.take(axes(X, 1), size)
        for j in axes(X, 2)
            X[i, j] = 0
        end
    end
    return X
end

Base.@propagate_inbounds function Manifolds._padpoint!(
    G::MultiAffine{<:Any,<:Any,size},
    q::AbstractMatrix,
) where {size}
    Manifolds._padvector!(G, q)
    for (i,j) in Base.Iterators.take(zip(axes(q)...), size)
        q[i,j] = 1
    end
    return q
end


function Manifolds.affine_matrix(
    G::MultiAffine,
    p
    )
    pis = submanifold_components(G, p)
    pmat = allocate_result(G, affine_matrix, pis...)
    map(copyto!, submanifold_components(G, pmat), pis)
    @inbounds Manifolds._padpoint!(G, pmat)
    return pmat
end

function Manifolds.affine_matrix(
    G::MultiAffine{TH, dim, size, 𝔽},
    ::Identity{MultiAffineOp{TH, dim, size, 𝔽}}
    ) where {TH, dim, size, 𝔽}
    n = _get_representation_dim(G)
    return LinearAlgebra.Diagonal{Float64}(LinearAlgebra.I, n)
end

function Manifolds.zero_vector(
    G::MultiAffine{TH, dim, size, 𝔽},
    ::Identity{MultiAffineOp{TH, dim, size, 𝔽}}
    ) where {TH, dim, size, 𝔽}
    res = allocate_result(G, typeof(zero_vector))
    fill!(res, 0.)
    return res
end


function Manifolds.screw_matrix(G::MultiAffine, X)
    Xis = submanifold_components(G, X)
    Xmat = allocate_result(G, screw_matrix, Xis...)
    map(copyto!, submanifold_components(G, Xmat), Xis)
    @inbounds Manifolds._padvector!(G, Xmat)
    return Xmat
end


Base.@propagate_inbounds function Manifolds.submanifold_component(
    ::MultiAffine{<:Any,dim,size},
    p::AbstractMatrix,
    ::Val{1},
    ) where {dim,size}
    return view(p,
                last(axes(p,1), dim),
                first(axes(p,2), size)
                )
end

Base.@propagate_inbounds function Manifolds.submanifold_component(
    ::MultiAffine{<:Any,dim},
    p::AbstractMatrix,
    ::Val{2},
    ) where {dim}
    return view(p,
                last(axes(p,1), dim),
                last(axes(p,2), dim)
                )
end

function Manifolds.submanifold_components(
    G::MultiAffine,
    p::AbstractMatrix,
    )
    d = _get_representation_dim(G)
    @assert Base.size(p) == (d,d)
    @inbounds t = submanifold_component(G, p, Val(1))
    @inbounds R = submanifold_component(G, p, Val(2))
    return (t, R)
end


function Manifolds._log_lie!(G::MultiAffine, X, q)
    qmat = affine_matrix(G, q)
    Xmat = real(Manifolds.log_safe(qmat))
    map(copyto!, submanifold_components(G, X), submanifold_components(G, Xmat))
    Manifolds._padvector!(G, X)
    return X
end

function Manifolds.exp_lie!(G::MultiAffine, q, X)
    Xmat = screw_matrix(G, X)
    qmat = exp(Xmat)
    map(copyto!, submanifold_components(G, q), submanifold_components(G, qmat))
    Manifolds._padpoint!(G, q)
    return q
end

# Alternative option: use the standard definition of adjoint_action
# should do it with matrices as well
# then could use this function for testing
function Manifolds.adjoint_action(G::MultiAffine, p, X)
    tmp = allocate_result(G, adjoint_action, X)
    return adjoint_action!(G, tmp, p, X)
end

Manifolds.adjoint_action!(::MultiAffine, Y, ::Identity, X) = copyto!(Y, X)

Manifolds.adjoint_action!(G::MultiAffine, tmp, p, X) = begin
    np, hp = submanifold_components(G, p)
    n, h = submanifold_components(G, tmp)
    nX, hX = submanifold_components(G, X)
    H = factor_group(G)
    adjoint_action!(H, h, hp, hX)
    A = G.op.action
    apply!(A, n, hp, nX)
    LinearAlgebra.axpy!(-1, apply_diff_group(A, Identity(H), h, np), n)
    return tmp
end


inverse_adjoint_action!(G::MultiAffine, Y, p, X) = adjoint_action!(G, Y, inv(G,p), X)

# Manifolds.translate_diff!(G::MultiAffine, Y, p, ::Any, X, dir::Tuple{RightAction, RightSide}) = inverse_adjoint_action!(G, Y, p, X)
Manifolds.translate_diff!(G::MultiAffine,
    Y, ::Any, ::Any, X,
    ::Manifolds.LeftForwardAction,
) = copyto!(G, Y, X)
Manifolds.translate_diff!(G::MultiAffine,
    Y, ::Any, ::Any, X,
    ::Manifolds.RightForwardAction,
) = copyto!(G, Y, X)
Manifolds.translate_diff!(
    G::MultiAffine,
    Y, p, ::Any, X,
    ::Manifolds.LeftBackwardAction,
) = adjoint_action!(G, Y, p, X)
Manifolds.translate_diff!(
    G::MultiAffine,
    Y, p, ::Any, X,
    ::Manifolds.RightBackwardAction,
) = inverse_adjoint_action!(G, Y, p, X)

# simply because it is defined for semi-direct products as well
Manifolds.translate_diff!(G::MultiAffine,
    Y, ::Identity, ::Any, X,
    # ::Manifolds.LeftForwardAction,
    ::Manifolds.LeftForwardAction,
) = copyto!(G, Y, X)
Manifolds.translate_diff!(G::MultiAffine,
    Y, ::Any, ::Identity, X,
    # ::Manifolds.LeftForwardAction,
    ::Manifolds.LeftForwardAction,
) = copyto!(G, Y, X)


Manifolds.inv_diff(::MultiAffine, ::Identity, ξ) = -ξ
Manifolds.inv_diff!(G::MultiAffine, Y, p, X) = -adjoint_action!(G, Y, p, X)


function Manifolds.lie_bracket(G::MultiAffine, v1, v2, )
    res = allocate_result(G, lie_bracket)
    return lie_bracket!(G, res, v1, v2)
end

function Manifolds.lie_bracket!(G::MultiAffine, res, v1, v2, )
    A = G.op.action
    n1, h1 = submanifold_components(v1)
    n2, h2 = submanifold_components(v2)
    rn = apply(A, h1, n2) - apply(A, h2, n1)
    rh = lie_bracket(factor_group(G), h1, h2)
    map(copyto!, submanifold_components(G, res), [rn,rh])
    return res
end

# The two morphisms corresponding to the sequence
# 0 -> V -> G -> H -> 0

function _fill_in!(G::MultiAffine, x, ts)
    X = submanifold_component(G, x, 1)
    copyto!(X, ts)
    return x
end

_fill_in!(G::MultiAffine, x, ts...) = _fill_in!(G, x, hcat(ts...))

factor_group(G::MultiAffine) = submanifold(G, 2)
normal_group(G::MultiAffine) = submanifold(G, 1)

from_normal_grp(G::MultiAffine, ts...) = begin
    x = identity_element(G)
    return _fill_in!(G, x, ts...)
end

from_normal_alg(G::MultiAffine, ts...) = begin
    x = zero_vector(G, identity_element(G))
    return _fill_in!(G, x, ts...)
end

to_factor(G::MultiAffine, pt) = submanifold_component(G,pt,2)

to_factor_grp(G::MultiAffine, pt) = to_factor(G, pt)

to_factor_alg(G::MultiAffine, pt) = to_factor(G, pt)

normal_indices(::MultiAffine{<:Any, dim}, idx; pos=0) where {dim} = collect(Iterators.take(Iterators.drop(idx, pos*dim), dim))
factor_indices(::MultiAffine{<:Any, dim, size}, idx) where {dim,size} = collect(Iterators.drop(idx, size * dim))
