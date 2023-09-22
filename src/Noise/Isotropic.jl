
"""
    IsotropicNoise(manifold, # metric manifold M
      deviation # function M -> 𝑹
    )


Model a noise with covariance equal to the metric at the given point on the manifold.
"""
struct IsotropicNoise{TM,TF<:Function} <: AbstractNoise
    manifold::TM # metric manifold M
    deviation::TF # function M -> [0, +∞)
end

sample_space(n::IsotropicNoise) = n.manifold

"""
    IsotropicNoise(M, std::Number)

Create an isotropic noise on the manifold `M`.
The number `std` is then the *standard deviation*,
which does not depend on the point of the manifold.
"""
IsotropicNoise(M, std::Number) = IsotropicNoise(M, ConstantFunction(std))

rescale_noise(n::IsotropicNoise, scale) = IsotropicNoise(n.manifold, x -> scale*n.deviation(x))

rescale_noise(n::IsotropicNoise{TA,TF}, scale) where{TA, TF<:ConstantFunction} = IsotropicNoise(n.manifold, scale*n.deviation)

get_basis_at(noise::IsotropicNoise, x) = get_basis(sample_space(noise), x, DefaultOrthonormalBasis())


function get_covariance_at(
    noise::IsotropicNoise,
    point,
    ::Union{CachedBasis{F, DefaultOrthonormalBasis{F, TV}, TD},
             DefaultOrthonormalBasis{F, TV}}
    ) where {F,TV,TD}
    dim = manifold_dimension(sample_space(noise))
    return PDMats.ScalMat(dim, noise.deviation(point)^2)
end



function add_noise(
    noise::IsotropicNoise,
    rng::Random.AbstractRNG,
    point
    )
    M = sample_space(noise)
    rv = isotropic_perturbation(rng, sample_space(noise), point)
    σ = noise.deviation(point)
    return exp(M, point, σ*rv)
end


"""
    isotropic_perturbation(rng, M, p)

Create noisy vector at the point `p` on the metric manifold `M`,
where the covariance matrix is the identity in an orthonormal basis.
"""
function isotropic_perturbation(
    rng::Random.AbstractRNG,
    M, # manifold
    p, # point on the manifold
    )
    dim = manifold_dimension(M)
    NT = ManifoldsBase.allocate_result_type(M, typeof(isotropic_perturbation), ())
    rc = randn(rng, NT, dim)
    B = DefaultOrthonormalBasis()
    rv = get_vector(M, p, rc, B)
    return rv
end

