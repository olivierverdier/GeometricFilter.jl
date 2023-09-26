import PDMats, LinearAlgebra, Random
"""
Encode a general covariance as a linear transformation A.
The covariance is then Σ = A*A'.
"""
struct Covariance{T, TM<:AbstractMatrix{T}} <: PDMats.AbstractPDMat{T}
    trafo::TM
end

# TODO: convert to PDMats or cholesky when trafo matrix becomes wide?

Covariance(M::LinearAlgebra.Symmetric{T,TM}) where {T,TM} = PDMats.chol_lower(cholesky(M))
Covariance(M::PDMats.PDMat) = Covariance(PDMats.chol_lower(M))
Covariance(M::PDMats.PDiagMat) = Covariance(sqrt.(M))

# Base.:*(M::AbstractMatrix, c::Covariance) = Covariance(M*c.trafo)

PDMats.unwhiten(c::Covariance, x::AbstractVecOrMat) = PDMats.unwhiten!(similar(x, get_dim(c)), c, x)
PDMats.unwhiten!(res::AbstractVecOrMat, c::Covariance, x::AbstractVecOrMat) = LinearAlgebra.mul!(res, c.trafo, x)

get_dim(c::Covariance) = size(c.trafo, 1)
get_mat(c::Covariance) = c.trafo*c.trafo'


### Conversion
Base.convert(::Type{Covariance{T}},         a::Covariance) where {T<:Real} = Covariance(convert(AbstractArray{T}, a.trafo))
Base.convert(::Type{AbstractArray{T}}, a::Covariance) where {T<:Real} = convert(Covariance{T}, a)
Base.convert(::Type{AbstractArray{T}}, a::Covariance{T}) where {T<:Real} = a

### Basics

Base.size(a::Covariance) = (get_dim(a), get_dim(a))
Base.Matrix(a::Covariance) = get_mat(a)
LinearAlgebra.diag(a::Covariance) = LinearAlgebra.diag(get_mat(a))
# LinearAlgebra.cholesky(a::PDMat) = a.chol

### Inheriting from AbstractMatrix

Base.getindex(a::Covariance, i::Int) = getindex(get_mat(a), i)
Base.getindex(a::Covariance, I::Vararg{Int, N}) where {N} = getindex(get_mat(a), I...)

### Arithmetics

Base.:+(a::Covariance, b::Covariance) = Covariance(hcat(a.trafo, b.trafo))
# function pdadd!(r::Matrix, a::Matrix, b::PDMat, c)
#     @check_argdims size(r) == size(a) == size(b)
#     PDMats._addscal!(r, a, get_mat(b), c)
# end

Base.:*(c::Real, a::Covariance) = Covariance(a.trafo * sqrt(c))
Base.:*(a::Covariance, c::Real) = c*a
Base.:*(a::Covariance, x::AbstractVector) = a.trafo*(a.trafo' * x)
Base.:*(a::Covariance, x::AbstractMatrix) = a.trafo * (a.trafo' * x)
# \(a::PDMat, x::AbstractVecOrMat) = a.chol \ x
# return matrix for 1-element vectors `x`, consistent with LinearAlgebra
# /(x::AbstractVecOrMat, a::PDMat) = reshape(x, Val(2)) / a.chol

### Algebra

# Base.inv(a::Covariance) = Covariance(inv(a.chol))
# LinearAlgebra.det(a::Covariance) = det(a.chol)
# LinearAlgebra.logdet(a::Covariance) = logdet(a.chol)
# LinearAlgebra.eigmax(a::Covariance) = eigmax(a.mat)
# LinearAlgebra.eigmin(a::Covariance) = eigmin(a.mat)
# Base.kron(A::Covariance, B::Covariance) = Covariance(kron(A.mat, B.mat), Cholesky(kron(A.chol.U, B.chol.U), 'U', A.chol.info))
# LinearAlgebra.sqrt(A::Covariance) = Covariance(sqrt(Hermitian(A.mat)))

### tri products

function PDMats.X_A_Xt(a::Covariance, x::AbstractMatrix)
    PDMats.@check_argdims get_dim(a) == size(x, 2)
    return Covariance(x * a.trafo)
end

PDMats.PDMat(a::Covariance) = a

# function Xt_A_X(a::PDMat, x::AbstractMatrix)
#     @check_argdims a.dim == size(x, 1)
#     z = chol_upper(a.chol) * x
#     return transpose(z) * z
# end

# function X_invA_Xt(a::PDMat, x::AbstractMatrix)
#     @check_argdims a.dim == size(x, 2)
#     z = x / chol_upper(a.chol)
#     return z * transpose(z)
# end

# function Xt_invA_X(a::PDMat, x::AbstractMatrix)
#     @check_argdims a.dim == size(x, 1)
#     z = chol_lower(a.chol) \ x
#     return transpose(z) * z
# end

depth(M::PDMats.AbstractPDMat) = M.dim
depth(M::Covariance) = size(M.trafo, 2)

sample(rng::Random.AbstractRNG, M::PDMats.AbstractPDMat) = PDMats.unwhiten(M, randn(rng, depth(M)))
