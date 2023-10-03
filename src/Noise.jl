

abstract type AbstractNoise end
#--------------------------------
# AbstractNoise Interface
#--------------------------------
"""
    sample_space(::AbstractNoise) :: AbstractManifold

Manifold on which the noise operates.
"""
function sample_space end

"""
    get_covariance_at(::AbstractNoise, x, ::AbstractBasis)  :: AbstractPDMat

A covariance matrix for this noise, at the given point,
 expressed in the given basis.
"""
function get_covariance_at end

"""
    get_basis_at(::AbstractNoise, x) :: AbstractBasis

The basis given by the noise model at the point x.
"""
function get_basis_at end
"""
    add_noise(::AbstractNoise, ::Random.AbstractRNG, x::TX) :: TX

Add noise at point x.

**Shortcut**: noise(rng, x) ≡ add_noise(noise, rng, x)
"""
function add_noise end

"""
    rescale_noise(TN<:AbstractNoise, λ) :: TN

Rescale the noise by a given factor ``λ``.
If ``X`` is the random variable distributed as the noise,
then return the distribution of ``λX``.

**Shortcut**: λ * noise ≡ rescale_noise(noise, λ)
"""
function rescale_noise end
#--------------------------------

#--------------------------------
# General Interface
#--------------------------------
Base.:*(s::Number, n::AbstractNoise) = rescale_noise(n, s)

(noise::AbstractNoise)(rng::Random.AbstractRNG, x) = add_noise(noise, rng, x)

get_basis_at(::AbstractNoise, ::Any) = DefaultOrthonormalBasis()

get_covariance_at(n::AbstractNoise, x) = get_covariance_at(n, x, get_basis_at(n, x))


#--------------------------------
# Specific Noises
#--------------------------------

include("Noise/Action.jl")
include("Noise/Isotropic.jl")
include("Noise/NoNoise.jl")




