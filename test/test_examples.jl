

@testset "Example Code $name" for name in [
    "flatfilter",
    # "imu",
    "localisation",
    "simple_loc",
    "testbed"]
    include("ex/ex_$name.jl")
end
