import Test
import Test: @testset

"""
A replacement to the @test macro
that shows the docstring of the failing function in case of failure.

Does not work with keywords (such as `broken`).
"""
macro test(ex)
    return quote
        Test.@testset "Custom test" begin
            result = $(esc(ex))
            Test.@test result
            if !result
                func = $(esc(ex.args[1]))
                func_module = parentmodule(func)
                if !(func_module in (Base, Core))
                    doc = Base.Docs.doc(func)
                    display(MIME("text/plain"), doc)
                    println()
                end
            end
        end
    end
end
