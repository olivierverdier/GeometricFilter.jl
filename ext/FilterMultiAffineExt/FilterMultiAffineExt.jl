# module FilterMultiAffineExt

# using GeometricFilter
using MultiAffine
using Manifolds

export AdjointLinearMotion
export PositionObserver

PositionObserver(A::MultiAffineAction{LeftAction, <:MultiAffineGroup{<:Any, dim}}) where {dim} = ActionObserver(A, zeros(dim))

include("AdjointLinear.jl")


# end
