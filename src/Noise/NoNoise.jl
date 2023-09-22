"""
    NoNoise(M::AbstractManifold)

Models the absence of noise on the given sample space `M`.
"""
struct NoNoise{TM} <: AbstractNoise
    manifold::TM
end

sample_space(n::NoNoise) = n.manifold

get_covariance_at(n::NoNoise, ::Any, ::Any) = PDMats.ScalMat(manifold_dimension(sample_space(n)), 0.)


add_noise(::NoNoise, ::Random.AbstractRNG, x) = x

rescale_noise(n::NoNoise, ::Any) = n
