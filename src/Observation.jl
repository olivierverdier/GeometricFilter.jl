
abstract type AbstractObservation end

@doc raw"""
    Observation(observer::Observer, noise::Noise, measurement)

Encapsulation of a full observation data, namely
- an observer (`AbstractObserver`)
- an observation noise (`AbstractNoise`)
- a measurement, i.e., a *point* lying on the observer's space
"""
struct Observation{TO,TN,TM} <: AbstractObservation
    observer::TO
    noise::TN
    measurement::TM
end

"""
    noisy_observation(rng::Random.AbstractRNG,
      obs::AbstractObserver,
       noise::AbstractNoise,
         state)

`Observation` object encapsulating a noisy version of `obs(state)`.
"""
noisy_observation(rng::Random.AbstractRNG, obs, noise, state) = Observation(obs, noise, noise(rng, obs(state)))

"""
    EmptyObservation()

An empty observation, that is, an observation
without any useful information to update
the state's uncertainty.
"""
struct EmptyObservation <: AbstractObservation
end

# So that AbstractObservation can be used in sparse vectors
Base.zero(::AbstractObservation) = EmptyObservation()
Base.zero(::Type{AbstractObservation}) = EmptyObservation()

Observation() = EmptyObservation()
