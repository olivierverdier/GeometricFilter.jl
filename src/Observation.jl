
abstract type AbstractObservation end

@doc raw"""
    Observation(observer::Observer, noise::Noise, measurement)

Encapsulation of a full observation data.
This includes the actual measurement, but also *how* it was measured
(the observer) and with how much noise.
The `Observation` object is thus built from
- an observer (`AbstractObserver`)
- an observation noise (`AbstractNoise`)
- a measurement, i.e., a *point* lying on the observer's space
"""
struct Observation{TO,TN,TM} <: AbstractObservation
    observer::TO
    noise::TN
    measurement::TM
    function Observation(observer, noise, measurement)
        M = observation_space(observer)
        if M != sample_space(noise)
            throw(ErrorException("Observation and noise manifolds should be the same, but\n\t$(M)\n\t≠\n\t$(sample_space(noise))"))
        end
        if !is_point(M, measurement)
            throw(ErrorException("Point should be on observation manifold"))
        end
        return new{typeof(observer), typeof(noise), typeof(measurement)}(observer, noise, measurement)
    end
end

Base.show(io::IO, obs::Observation) = print(io, "Observation($(obs.observer), $(obs.noise), $(obs.measurement))")

"""
    noisy_observation(rng::Random.AbstractRNG,
      obs::AbstractObserver,
       noise::AbstractNoise,
         state)

`Observation` object encapsulating a noisy version of `obs(state)`.
"""
noisy_observation(rng::Random.AbstractRNG, obs, noise, state) = Observation(obs, noise, noise(rng, obs(state)))

function simulate_observations(flag, rng, states, observers, noises)
    observations = map(enumerate(zip(states, observers, noises))) do (i, (state, observer, noise))
        if flag(i)
            return noisy_observation(rng, observer, noise, state)
        else
            return Observation()
        end
    end |> SparseVector
    return observations
end

"""
    EmptyObservation()

An empty observation, that is, an observation
without any useful information to update
the state's uncertainty.
"""
struct EmptyObservation <: AbstractObservation
end

Base.show(io::IO, ::EmptyObservation) = print(io, "Observation()")

# So that AbstractObservation can be used in sparse vectors
Base.zero(::AbstractObservation) = EmptyObservation()
Base.zero(::Type{AbstractObservation}) = EmptyObservation()
Base.zero(::Type{EmptyObservation}) = EmptyObservation()

Observation() = EmptyObservation()
