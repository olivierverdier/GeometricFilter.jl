using Test
using MultiAffine
import ManifoldGroupUtils as GU
import ManifoldGroupUtils: rand_lie
import PDMatsSingular

test_files = [
    "test_filter_measure.jl",
    "test_motion.jl",
    "test_noise.jl",
    "test_filter.jl",
    "test_generic_filter.jl",
    "test_observer.jl",
    "test_projlognormal.jl",
    "test_stochastic_motion.jl",

    # "test_examples.jl",
    # "test_ex_inertial.jl",
]

@time @testset " " for path in test_files
    printstyled("─"^16 * "[ $path ]\n"; color=:yellow)
    @time include(path)
end

# "✓"

