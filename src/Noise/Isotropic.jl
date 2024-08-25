
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

Base.show(io::IO, n::IsotropicNoise) = print(io, "IsotropicNoise($(n.manifold), $(n.deviation))")

sample_space(n::IsotropicNoise) = n.manifold

"""
    IsotropicNoise(M, std::Number)

Create an isotropic noise on the manifold `M`.
The number `std` is then the *standard deviation*,
which does not depend on the point of the manifold.
"""
IsotropicNoise(M, std::Number) = IsotropicNoise(M, ConstantFunction(std))

rescale_noise(n::IsotropicNoise, scale) = IsotropicNoise(n.manifold, x -> scale*n.deviation(x))

rescale_noise(n::IsotropicNoise{<:Any,TF}, scale) where{TF<:ConstantFunction} = IsotropicNoise(n.manifold, scale*n.deviation)


function get_covariance_at(
    noise::IsotropicNoise,
    point,
    ::Union{CachedBasis{𝔽,DefaultOrthonormalBasis{𝔽}},
             DefaultOrthonormalBasis}
) where {𝔽}
    dim = manifold_dimension(sample_space(noise))
    return PDMats.ScalMat(dim, noise.deviation(point)^2)
end



function add_noise(
    noise::IsotropicNoise,
    rng::Random.AbstractRNG,
    point
    )
    M = plain_manifold(sample_space(noise))
    rv = isotropic_perturbation(rng, M, point)
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

