using Test

test_files = [
    "aux/Manifolds/test_multiaffine.jl",
    "aux/Manifolds/test_multiaffineaction.jl",
    "aux/PDMats/test_covariance.jl",
    "test_motion.jl",
    "test_noise.jl",
    "test_filter.jl",
    "test_generic_filter.jl",
    "test_observer.jl",
    "test_projlognormal.jl",
    "test_grouptools.jl",
    "test_filter_measure.jl",
    "test_examples.jl",
    "test_stochastic_motion.jl",
    "test_ex_inertial.jl",
]

@time @testset " " for path in test_files
    printstyled("─"^16 * "[ $path ]\n"; color=:yellow)
    @time include(path)
end

# "✓"

