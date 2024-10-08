

"""
    predict(distribution::AbstractActionDistribution, stochastic_motion::StochasticMotion)

Compute the update of the uncertainty (the distribution),
given a motion and its associated process noise (a stochastic motion).
"""
predict(
    distribution::AbstractActionDistribution,
    stochastic_motion
)

predict(distribution::AbstractActionDistribution{TA}, sm::AbstractStochasticMotion{TA}) where {TA} = _predict(distribution, sm.motion, sm.noise)
@deprecate predict(distribution, motion, noise) predict(distribution, StochasticMotion(motion, noise))
predict(d::AbstractActionDistribution{TA}, m::AbstractMotion{TA}) where {TA} = _predict(d, m)

#--------------------------------
# Prediction Helpers
#--------------------------------
"""
    _predict(distribution::ActionDistribution, motion::Motion, process_noise::ActionNoise) :: ActionDistribution

Updates the distribution according to the motion and the process noise.
"""
function _predict(
    distribution::AbstractActionDistribution{TA}, # filtering distribution
    motion::AbstractMotion{TA}, # abstract model with the relevant motion
    process_noise=nothing;
    dt=0.1 # discretisation time step
) where {TA}
    assert_equal_properties(distribution, ManifoldNormal.get_action, motion, AffineMotions.get_action, "Different distribution and motion actions")
    x = Distributions.mean(distribution)

    χ, morph = AffineMotions.compute_morphism(motion, x, ManifoldNormal.get_lie_basis(distribution); dt=dt)

    x_ = apply(AffineMotions.get_action(motion), χ, x)

    Σ = Distributions.cov(distribution)
    if isnothing(process_noise)
        Σ_ = Σ
    else
        Σ_ = Σ + ManifoldNormal.get_lie_covariance_at(process_noise, x_, ManifoldNormal.get_lie_basis(distribution))
    end
    Σ__ = PDMats.X_A_Xt(Σ_, morph)
    return update_mean_cov(distribution, x_, PDMats.AbstractPDMat(Σ__))
end

"""
    add_process_noise(dist::AbstractActionDistribution, noise::AbstractActionNoise)

[Deprecated]

Add process noise to a distribution of type `AbstractActionDistribution`.
"""
function add_process_noise(
    distribution::AbstractActionDistribution,
    process_noise,
)
    x = Distributions.mean(distribution)
    Σ = Distributions.cov(distribution)
    Σ_ = PDMats.AbstractPDMat(Σ + get_lie_covariance_at(process_noise, x, get_lie_basis(distribution)))
    return update_mean_cov(distribution, x, Σ_)
end
