
"""
    NoiseMode

Whether the noise is already encoded in the motion (typical for actual measurements), or as a noise applied on the resulting position.
"""
abstract type NoiseMode end

struct MotionMode <: NoiseMode end
struct PositionMode <: NoiseMode end

abstract type AbstractStochasticMotion{TNM, TA<:AbstractGroupAction{LeftAction}} end

#--------------------------------
# AbstractStochasticMotion Interface
#--------------------------------
"""
    get_motion(m::AbstractStochasticMotion) :: AbstractMotion

The underlying deterministic motion.
"""
function get_motion end

"""
    get_noise(m::AbstractStochasticMotion) :: AbstractActionNoise

The underlying process noise.
"""
function get_noise end

"""
    apply_noise(::RNG, ::StochasticMotion, dist::ProjLogNormal)

In `PositionMode`, apply noise to the mean of the distribution `dist`.
This models a noisy measurement of the motion.
In `MotionMode`, the motions are supposed to be already marred by
uncertainty, so this does not do anything.
"""
function apply_noise end
#--------------------------------

struct StochasticMotion{TNM,TM,TN,TA} <: AbstractStochasticMotion{TNM,TA}
    motion::TM
    noise::TN
end

Base.show(io::IO, sm::StochasticMotion{TNM}) where {TNM} = print(io, "StochasticMotion($(sm.motion), $(sm.noise), $TNM())")

@doc raw"""
    StochasticMotion(motion::Motion, noise::Noise, mode=PositionMode())

Encapsulate the idea of a stochastic dynamical system on a manifold ``\mathcal{M}``, defined by a motion ``φ \colon \mathcal{M}→\mathfrak{g}``, and a noise model on the manifold.
"""
StochasticMotion(motion::AbstractMotion{TA}, noise::AbstractActionNoise{TA}, mode::NoiseMode=PositionMode()) where {TA} = StochasticMotion{typeof(mode),typeof(motion),typeof(noise), TA}(motion, noise)

get_motion(s::StochasticMotion) = s.motion
get_noise(s::StochasticMotion) = s.noise

"""
    integrate(rng::RNG, s::StochasticMotion, x)

Integrate the stochastic motion: deterministically integrate
the underlying motion, and adds noise on the result.

In `PositionMode`, the noise is added after exact integration.
In `MotionMode`, the underlying motions are already noisy.
"""
# integrate(rng::Random.AbstractRNG, s::AbstractStochasticMotion{PositionMode}, x) = get_noise(s)(rng, integrate(get_motion(s), x))
function integrate(rng::Random.AbstractRNG, s::AbstractStochasticMotion{PositionMode}, x)
    return get_noise(s)(rng, integrate(get_motion(s), x))
end
integrate(::Random.AbstractRNG, s::AbstractStochasticMotion{MotionMode}, x) = integrate(get_motion(s), x)

apply_noise(rng::Random.AbstractRNG, sm::AbstractStochasticMotion{PositionMode}, dist) = update_mean(dist, get_noise(sm)(rng, Distributions.mean(dist)))
apply_noise(::Random.AbstractRNG, ::AbstractStochasticMotion{MotionMode}, dist) = dist
