abstract type ManifoldVariate{TM} <: Distributions.VariateForm end

abstract type AbstractProjLogNormal{TA<:AbstractGroupAction{LeftAction}} <: Distributions.Sampleable{ManifoldVariate{AbstractManifold}, Distributions.Continuous} end

#--------------------------------
# AbstractProjLogNormal Interface
#--------------------------------
"""
    get_action(d::AbstractProjLogNormal) :: AbstractGroupAction

The underlying action (i.e., the underlying homogeneous space).
"""
function get_action end

"""
    Distribution.mean(d::AbstractProjLogNormal) :: [element of manifold M]
    Distribution.cov(d::AbstractProjLogNormal) :: [covariance in Alg(G)]

The mean and covariance of the distribution.
"""

@doc raw"""
    get_lie_basis(d::AbstractProjLogNormal) :: Basis

The basis of ``\mathfrak{g}`` in which the covariance is defined.
"""
function get_lie_basis end

"""
    ProjLogNormal(action::Action, μ, Σ::PDMat, B::Basis)

Wrapped exponential distribution on the space of the given action.
"""
struct ProjLogNormal{TA<:AbstractGroupAction{LeftAction},TM,TN<:PDMats.AbstractPDMat,TB} <: AbstractProjLogNormal{TA}
    action::TA # left group action G ⊂ Diff(M)
    μ::TM # mean: element of M
    Σ::TN # centred normal distribution on Alg(G) (in basis B)
    B::TB # basis of Alg(G)
    function ProjLogNormal(action::TA, μ::TM, Σ::TN, B::TB) where {TA,TM,TN, TB}
        @assert is_point(group_manifold(action), μ)
        return new{TA,TM,TN,TB}(action, μ, Σ, B)
    end
end

Base.show(io::IO, dist::ProjLogNormal) = print(io, "ProjLogNormal($(dist.action), $(dist.μ), $(dist.Σ), $(dist.B))")

function ProjLogNormal(
    A, # action
    x, # in M
    σ :: Number, # isotropic variance
    B :: AbstractBasis
    )
    G = base_group(A)
    dim = manifold_dimension(G)
    Σ = PDMats.ScalMat(dim, σ)
    return ProjLogNormal(A, x, Σ, B)
end

ProjLogNormal(A, x, σ::Number) = ProjLogNormal(A, x, σ, DefaultOrthonormalBasis())
ProjLogNormal(A, x, Σ::PDMats.AbstractPDMat) = ProjLogNormal(A, x, Σ, DefaultOrthonormalBasis())


Distributions.cov(d::ProjLogNormal) = d.Σ
Distributions.mean(d::ProjLogNormal) = d.μ
get_action(d::ProjLogNormal) = d.action
get_lie_basis(d::ProjLogNormal) = d.B

update_mean_cov(d::ProjLogNormal{TA,TM,TN,TB}, μ::TM, Σ) where {TA,TM,TN,TB} = ProjLogNormal(d.action, μ, Σ, d.B)

update_mean(d::ProjLogNormal, x) = update_mean_cov(d, x, Distributions.cov(d))

function Base.length(d::ProjLogNormal)
    M = group_manifold(d.action)
    return manifold_dimension(M)
end

function rand!(
    rng::Random.AbstractRNG,
    d::AbstractProjLogNormal,
    out::AbstractArray{F},
    ) where {F}
    rc = sample(rng, Distributions.cov(d))
    G = base_group(d.action)
    ξ = get_vector_lie(G, rc, d.B)
    χ = exp_lie(G, ξ)
    apply!(d.action, out, χ, Distributions.mean(d))
    return out
end

function Base.rand(
    rng::Random.AbstractRNG,
    d::AbstractProjLogNormal,
    )
    M = group_manifold(d.action)
    x = allocate_result(M, typeof(rand))
    rand!(rng, d, x)
    return x
end

"""
    action_noise(D::ProjLogNormal) :: ActionNoise

Create an action noise object from a ProjLogNormal distribution.
This is simply an action noise with constant covariance.
"""
action_noise(D::AbstractProjLogNormal) = ActionNoise(get_action(D), ConstantFunction(D.Σ), get_lie_basis(D))


@doc raw"""
    scaled_distance(D::ProjLogNormal, x)

Start with a `ProjLogNormal(μ,Σ)` distribution with mean ``μ`` lying
on the sample manifold ``M``, and
covariance ``Σ`` in the Lie algebra ``\mathfrak{g}`` of the group ``G``.
The infinitesimal action of the group ``G``
on the manifold ``M`` gives rise to the linear map ``P \colon \mathfrak{g} \to T_{μ}M``,
which in turns gives the projected covariance ``PΣP^*``
on the tangent space ``T_{μ}M``.

One then measures the scaled distance from
``μ`` to ``x`` from ``v := \log(μ, x)`` with the formula
```math
\frac{\sqrt{v^T (PΣP^*)^{-1} v}}{\sqrt{n}}
```
(where ``v`` is regarded as a column matrix here)
and ``n`` is the dimension of the sample manifold ``M``.
"""
function scaled_distance(D::AbstractProjLogNormal, x)
    noise = action_noise(D)
    B = DefaultOrthonormalBasis()
    x0 = Distributions.mean(D)
    mat = get_covariance_at(noise, x0, B)
    M = sample_space(noise)
    vel = log(M, x0, x)
    vc = get_coordinates(M, x0, vel, B)
    vc_ = reshape(vc, :, 1)
    return sqrt(first(PDMats.Xt_A_X(mat, vc_))/manifold_dimension(M))
end

