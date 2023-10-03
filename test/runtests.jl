
function include_test(path)
    @info "Testing $path"
    @time include(path)  # show basic timing, (this will print a newline at end)
end

include_test("aux/Manifolds/test_multiaffine.jl")
include_test("aux/Manifolds/test_multiaffineaction.jl")
include_test("aux/PDMats/test_covariance.jl")
include_test("test_motion.jl")
include_test("test_noise.jl")
include_test("test_filter.jl")
include_test("test_generic_filter.jl")
include_test("test_observer.jl")
include_test("test_projlognormal.jl")
include_test("test_grouptools.jl")
include_test("test_filter_measure.jl")
include_test("test_examples.jl")
include_test("test_stochastic_motion.jl")



