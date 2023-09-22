



const MultiColumnwiseMultiplicationAction{G,dim,size,𝔽} = Manifolds.ColumnwiseMultiplicationAction{
    # MultiTranslation{dim,size,𝔽},
    # Euclidean{Tuple{dim,size},𝔽},
    TranslationGroup{Tuple{dim,size},𝔽},
    G,
    LeftAction
}

const MultiAffine{G,dim,size,𝔽} = SemidirectProductGroup{
    𝔽,
    # MultiTranslation{dim,size,𝔽},
    TranslationGroup{Tuple{dim,size},𝔽},
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
function MultiAffine(G::Manifolds.GeneralUnitaryMultiplicationGroup{dim,𝔽,S}, size::Integer=1) where {dim, 𝔽, S}
    # where {dim<:Integer,𝔽<:AbstractNumbers, S}
    # space = PowerGroup(PowerManifold(TranslationGroup(dim; field=𝔽), size))
    # space = Euclidean(dim,size; field=𝔽)
    space = TranslationGroup(dim,size;field=𝔽)
    action = Manifolds.ColumnwiseMultiplicationAction(space, G)
    group = GroupManifold(ProductManifold(space, G), Manifolds.SemidirectProductOperation(action))
    return group
end


Base.show(io::IO, ::MultiAffine{G, dim,size, F}) where {G, dim,size, F} = print(io, "MultiAffine($(G),$(size))")

_get_representation_dim(G::MultiAffine{TH,dim,size,𝔽}
                        ) where {TH,dim,size,𝔽} = dim+size


function Manifolds.allocate_result(G::MultiAffine{TH,dim,size,𝔽}, ::Union{typeof(affine_matrix),typeof(screw_matrix)}, Xis...) where {TH,dim,size,𝔽}
    d = _get_representation_dim(G)
    return allocate(Xis[1], Manifolds.Size(d,d))
end

Base.@propagate_inbounds function Manifolds._padvector!(
    ::MultiAffine{TH,dim,size,𝔽},
    X::AbstractMatrix,
) where {TH,dim,size,𝔽}
    for i in Base.Iterators.take(axes(X, 1), size)
        for j in axes(X, 2)
            X[i, j] = 0
        end
    end
    return X
end

Base.@propagate_inbounds function Manifolds._padpoint!(
    G::MultiAffine{TH,dim,size,𝔽},
    q::AbstractMatrix,
) where {TH,dim,size,𝔽}
    Manifolds._padvector!(G, q)
    for (i,j) in Base.Iterators.take(zip(axes(q)...), size)
        q[i,j] = 1
    end
    return q
end


function Manifolds.affine_matrix(
    G::MultiAffine{TH,dim,size,𝔽},
    p
    ) where {TH,dim,size,𝔽}
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


function Manifolds.screw_matrix(G::MultiAffine{TH, dim, size, 𝔽}, X) where {TH, dim, size, 𝔽}
    Xis = submanifold_components(G, X)
    Xmat = allocate_result(G, screw_matrix, Xis...)
    map(copyto!, submanifold_components(G, Xmat), Xis)
    @inbounds Manifolds._padvector!(G, Xmat)
    return Xmat
end


Base.@propagate_inbounds function Manifolds.submanifold_component(
    ::MultiAffine{TH,dim,size,𝔽},
    p::AbstractMatrix,
    ::Val{1},
    ) where {TH, dim,size, 𝔽}
    return view(p,
                last(axes(p,1), dim),
                first(axes(p,2), size)
                )
end

Base.@propagate_inbounds function Manifolds.submanifold_component(
    ::MultiAffine{TH,dim,size,𝔽},
    p::AbstractMatrix,
    ::Val{2},
    ) where {TH, dim, size, 𝔽}
    return view(p,
                last(axes(p,1), dim),
                last(axes(p,2), dim)
                )
end

function Manifolds.submanifold_components(
    G::MultiAffine{TH,dim,size,𝔽},
    p::AbstractMatrix,
    ) where {TH,dim,size,𝔽}
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
function Manifolds.adjoint_action(G::MultiAffine{TH,dim,size,𝔽}, p, X) where {TH,dim,size,𝔽}
    tmp = allocate_result(G, adjoint_action, X)
    return adjoint_action!(G, tmp, p, X)
end

function Manifolds.adjoint_action!(G::MultiAffine{TH,dim,size,𝔽}, tmp, p, X) where {TH,dim,size,𝔽}
    mat = affine_matrix(G, p)
    matinv = affine_matrix(G, inv(G,p))
    res = mat * screw_matrix(G,X) * matinv
    map(copyto!, submanifold_components(G, tmp), submanifold_components(G, res))
    return tmp
end


function Manifolds.apply_diff(A::Manifolds.ColumnwiseMultiplicationAction{N,F,LeftAction}, a, ::Any, X) where {N,F}
    return apply(A, a, X)
end




function Manifolds.translate_diff!(G::MultiAffine{TH,dim,size,𝔽}, Y, p, q, X, dir::RightAction) where {TH,dim,size,𝔽}
    np, hp = submanifold_components(G, p)
    nq, hq = submanifold_components(G, q)
    nX, hX = submanifold_components(G, X)
    nY, hY = submanifold_components(G, Y)
    H = submanifold(G, 2)
    hY .= translate_diff!(H, hY, hp, hq, hX, dir)
    copyto!(nY, hq * (hX * np) + nX)
    @inbounds Manifolds._padvector!(G, Y)
    return Y
end



function Manifolds.lie_bracket(G::MultiAffine{TH,dim,size,𝔽}, v1, v2, ) where {TH,dim,size,𝔽}
    res = allocate_result(G, lie_bracket)
    return lie_bracket!(G, res, v1, v2)
end

function Manifolds.lie_bracket!(G::MultiAffine{TH,dim,size,𝔽}, res, v1, v2, ) where {TH,dim,size,𝔽}
    A = G.op.action
    n1, h1 = submanifold_components(v1)
    n2, h2 = submanifold_components(v2)
    rn = apply(A, h1, n2) - apply(A, h2, n1)
    rh = lie_bracket(G.manifold.manifolds[2], h1, h2)
    map(copyto!, submanifold_components(G, res), [rn,rh])
    return res
end

# The two morphisms corresponding to the sequence
# 0 -> V -> G -> H -> 0

# fix this stupid use of multiple dispatch; only allow lists of columns?
function _fill_in(G::MultiAffine{TH,dim,size,𝔽},
                  ts::AbstractArray{T,1}, x) where {TH,dim,size,𝔽,T}
    mat = submanifold_component(G, x, 1)
    map(copyto!, eachcol(mat), ts)
    return x
end

function _fill_in(G::MultiAffine{TH,dim,size,𝔽},
                  ts::AbstractArray{T,2}, x) where {TH,dim,size,𝔽,T}
    return _fill_in(G, eachcol(ts), x)
end

function from_normal_grp(G::MultiAffine,
                         ts::AbstractArray)
    x = identity_element(G)
    return _fill_in(G, ts, x)
end


function from_normal_alg(G::MultiAffine,
                         ts::AbstractArray)
    x = zero_vector(G, identity_element(G))
    return _fill_in(G, ts, x)
end

function to_factor(G::MultiAffine, pt)
    return submanifold_component(G,pt,2)
end

function to_factor_grp(G::MultiAffine, pt)
    return to_factor(G, pt)
end

function to_factor_alg(G::MultiAffine, pt)
    return to_factor(G, pt)
end

# action on affine space?

# function apply_diff_group(
#     A::GroupOperationAction{MultiDisplacement{dim,size},LeftAction},
#     id::Identity,
#     X,
#     p
# ) where {dim,size}
#     G = base_group(A)
#     res = allocate_result(G, apply_diff_group)
#     return apply_diff_group!(A,res,id,X,p)
# end


# function apply_diff_group!(
#     A::GroupOperationAction{MultiDisplacement{dim,size},LeftAction},
#     res,
#     ::Identity,
#     X,
#     p
# ) where {dim,size}
#     G = base_group(A)
#     mat_res = screw_matrix(G, X) * affine_matrix(G, p)
#     # folloowing is wrong, because tangent vector at H is stored as an element in the Lie algebra
#     map(copyto!, submanifold_components(G, res), submanifold_components(G, mat_res))
#     return res
# end
