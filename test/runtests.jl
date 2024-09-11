include("header.jl")

using MultiAffine
import ManifoldGroupUtils as GU
import ManifoldGroupUtils: rand_lie
import PDMatsSingular
using AffineMotions
using ManifoldNormal

test_files = [
    "test_filter_measure.jl",
    "test_filter.jl",
    "test_generic_filter.jl",
    "test_observer.jl",
    "test_stochastic_motion.jl",

    # "test_ex_inertial.jl",
    "test_observation.jl",
    "test_examples.jl",
]

@time @testset " " for path in test_files
    printstyled("─"^16 * "[ $path ]\n"; color=:yellow)
    @time include(path)
end

# "✓"

