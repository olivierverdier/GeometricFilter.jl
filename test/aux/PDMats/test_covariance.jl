

using Test
using GeometricFilter

@testset "Add" begin
    c = Covariance([1; 0;;])
    M = zeros(2,2)
    M+c
    c+M
end
