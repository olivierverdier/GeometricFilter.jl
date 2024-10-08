

abstract type AbstractStochasticMotion{TA<:AbstractGroupAction{LeftAction}} end

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

#--------------------------------

struct StochasticMotion{TM,TN,TA} <: AbstractStochasticMotion{TA}
    motion::TM
    noise::TN
end

Base.show(io::IO, sm::StochasticMotion{TNM}) where {TNM} = print(io, "StochasticMotion($(sm.motion), $(sm.noise), $TNM())")

@doc raw"""
    StochasticMotion(motion::Motion, process_noise::ActionNoise)

Encapsulate the idea of a stochastic dynamical system on a manifold ``\mathcal{M}``, defined by a motion ``φ \colon \mathcal{M}→\mathfrak{g}``, and a noise model on the manifold.

The noise `process_noise` must implement `get_lie_covariance_at`, so must be of type `AbstractActionNoise`.
"""
StochasticMotion(motion::AbstractMotion{TA}, noise::AbstractActionNoise{TA}) where {TA} = StochasticMotion{typeof(motion),typeof(noise),TA}(motion, noise)

get_motion(s::StochasticMotion) = s.motion
get_noise(s::StochasticMotion) = s.noise

"""
    sensor_perturbation(rng::RNG, sm::StochasticMotion, x) :: StochasticMotion

Simulate sensor noise by adding a random rigid motion with
velocity drawn from the stochastic motions noise.
"""
sensor_perturbation(rng::Random.AbstractRNG, sm::AbstractStochasticMotion, x) = StochasticMotion(get_motion(sm) + rigid_perturbation(rng, get_noise(sm), x), get_noise(sm))

"""
    simulate(s::StochasticMotion, x::TM,
      ::FilteringMode,
      ) ::TM

Integrate the stochastic motion: deterministically integrate
the underlying motion, and adds noise on the result.

The `FilterMode` can be the following:
- With `DataMode`, the motion is integrated exactly.
- With `PositionPerturbation(rng)`, the noise is added after exact integration.
- With `SensorPerturbation(rng)`, the noise is applied to the motion, followed by an exact integration.
"""
simulate(s::AbstractStochasticMotion, x, fm::FilteringMode)

simulate(s::AbstractStochasticMotion, x, ::DataMode) = integrate(get_motion(s), x)
simulate(s::AbstractStochasticMotion, x, fm::SimulationMode{PositionPerturbationMode}) = get_noise(s)(fm.rng, simulate(s, x, DataMode()))
simulate(s::AbstractStochasticMotion, x, fm::SimulationMode{SensorPerturbationMode}) = simulate(sensor_perturbation(fm.rng, s, x), x, DataMode())


"""
    rigid_perturbation(rng::RNG, noise::ActionNoise, x) :: RigidMotion

Compute a random rigid motion using the covariance of the
stochastic motion at ``x``, where ``x`` is a point on the sample
space of the noise.
"""
function rigid_perturbation(rng::Random.AbstractRNG, pnoise::AbstractActionNoise, x)
    action = pnoise.action
    Σ = ManifoldNormal.get_lie_covariance_at(pnoise, x)
    vec = sample(rng, Σ)
    ξ = get_vector_lie(base_group(action), vec, pnoise.basis)
    return RigidMotion(action, ξ)
end
