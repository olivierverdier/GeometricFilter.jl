

"""
    predict(distribution::AbstractProjLogNormal, stochastic_motion::StochasticMotion)

Compute the update of the uncertainty (the distribution),
given a motion and its associated process noise (a stochastic motion).
"""
predict(
    distribution::AbstractProjLogNormal,
    stochastic_motion
)

predict(distribution::AbstractProjLogNormal{TA}, sm::AbstractStochasticMotion{TA}) where {TA} = _predict(distribution, sm.motion, sm.noise)
@deprecate predict(distribution, motion, noise) predict(distribution, StochasticMotion(motion, noise))
predict(d::AbstractProjLogNormal{TA}, m::AbstractMotion{TA}) where {TA} = _predict(d, m)

#--------------------------------
# Prediction Helpers
#--------------------------------
"""
    _predict(distribution::ProjLogNormal, motion::Motion, process_noise::ActionNoise) :: ProjLogNormal

Updates the distribution according to the motion and the process noise.
"""
function _predict(
    distribution::AbstractProjLogNormal{TA}, # filtering distribution
    motion::AbstractMotion{TA}, # abstract model with the relevant motion
    process_noise=nothing;
    dt=0.1 # discretisation time step
) where {TA}
    assert_equal_actions(distribution, motion, "Different distribution and motion actions")
    x = Distributions.mean(distribution)

    χ, morph = compute_morphism(motion, x, get_lie_basis(distribution); dt=dt)

    x_ = apply(get_action(motion), χ, x)

    Σ = Distributions.cov(distribution)
    if process_noise === nothing
        Σ_ = Σ
    else
        Σ_ = Σ + get_lie_covariance_at(process_noise, x_, get_lie_basis(distribution))
    end
    Σ__ = PDMats.X_A_Xt(Σ_, morph)
    return update_mean_cov(distribution, x_, PDMats.AbstractPDMat(Σ__))
end

"""
    add_process_noise(dist::AbstractProjLogNormal, noise::AbstractActionNoise)

[Deprecated]

Add process noise to a distribution of type `AbstractProjLogNormal`.
"""
function add_process_noise(
    distribution::AbstractProjLogNormal,
    process_noise,
)
    x = Distributions.mean(distribution)
    Σ = Distributions.cov(distribution)
    Σ_ = PDMats.AbstractPDMat(Σ + get_lie_covariance_at(process_noise, x, get_lie_basis(distribution)))
    return update_mean_cov(distribution, x, Σ_)
end
