
@testset "$name" for name in [
    "flatfilter",
    "inertial",
    "localisation",
    "simple_loc",
    "testbed"]
    include("ex/ex_$name.jl")
end
