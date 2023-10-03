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
struct ProjLogNormal{TA<:AbstractGroupAction{LeftAction},TM,TN} <: AbstractProjLogNormal{TA}
    μ::TM # mean: element of M
    noise::TN
    function ProjLogNormal(μ::TM, noise::TN) where {TM,TN}
        @assert is_point(sample_space(noise), μ)
        return new{typeof(noise.action), TM, TN}(μ, noise)
    end
end

Base.show(io::IO, dist::ProjLogNormal) = print(io, "ProjLogNormal($(dist.μ), $(dist.noise))")

ProjLogNormal(action, μ, Σ, B=DefaultOrthonormalBasis()) = ProjLogNormal(μ, ActionNoise(action, PDMats.AbstractPDMat(Σ), B))

function ProjLogNormal(
    A, # action
    x, # in M
    σ :: Number, # isotropic variance
    B=DefaultOrthonormalBasis() :: AbstractBasis
    )
    G = base_group(A)
    dim = manifold_dimension(G)
    Σ = PDMats.ScalMat(dim, σ)
    return ProjLogNormal(A, x, Σ, B)
end


Distributions.cov(d::ProjLogNormal) = d.noise.covariance()
Distributions.mean(d::ProjLogNormal) = d.μ
get_action(d::ProjLogNormal) = d.noise.action
get_lie_basis(d::ProjLogNormal) = d.noise.basis

"""
    update_mean_cov(d::ProjLogNormal, μ, Σ)

Return new `ProjLogNormal` object with new mean ``μ`` and covariance ``Σ``.
"""
update_mean_cov(d::ProjLogNormal{<:Any,TM}, μ::TM, Σ) where {TM}  = ProjLogNormal(μ, update_cov(d.noise, Σ))

"""
    update_mean_cov(d::ProjLogNormal, μ)

Return new `ProjLogNormal` object with new mean ``μ``.
"""
update_mean(d::ProjLogNormal, x) = update_mean_cov(d, x, Distributions.cov(d))

function Base.length(d::ProjLogNormal)
    M = group_manifold(get_action(d))
    return manifold_dimension(M)
end

function rand!(
    rng::Random.AbstractRNG,
    d::AbstractProjLogNormal,
    out::AbstractArray,
    ) 
    rc = sample(rng, Distributions.cov(d))
    G = base_group(get_action(d))
    ξ = get_vector_lie(G, rc, get_lie_basis(d))
    χ = exp_lie(G, ξ)
    apply!(get_action(d), out, χ, Distributions.mean(d))
    return out
end

function Base.rand(
    rng::Random.AbstractRNG,
    d::AbstractProjLogNormal,
    )
    M = sample_space(d.noise)
    x = allocate_result(M, typeof(rand))
    rand!(rng, d, x)
    return x
end

"""
    action_noise(D::ProjLogNormal) :: ActionNoise

Create an action noise object from a ProjLogNormal distribution.
This is simply an action noise with constant covariance.
"""
action_noise(D::AbstractProjLogNormal) = D.noise


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

