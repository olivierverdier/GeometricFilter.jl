abstract type FilteringMode end

struct DataMode <: FilteringMode end

struct SimulationMode{T,TRNG} <: FilteringMode
    rng::TRNG
end

abstract type PerturbationType  end
struct SensorPerturbationMode <: PerturbationType end
struct PositionPerturbationMode <: PerturbationType end

SimulationMode(rng::Random.AbstractRNG, t::PerturbationType) = SimulationMode{typeof(t),typeof(rng)}(rng)

PositionPerturbation(rng) = SimulationMode(rng, PositionPerturbationMode())
SensorPerturbation(rng) = SimulationMode(rng, SensorPerturbationMode())
