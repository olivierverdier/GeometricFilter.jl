

function run_script_in_isolation(script_path)
    mod = Module()

    # Define a custom include function for the new module
    custom_include(path) = Base.include(mod, path)

    # Add the custom include function to the new module
    Core.eval(mod, :(include(path) = $custom_include(path)))

    # Run the script in the new module
    result = Base.include(mod, script_path)

    return mod, result
end


@time @testset "$name" for name in [
    "flatfilter",
    "inertial",
    "localisation",
    "simple_loc",
    "testbed",
    ]
    printstyled("─"^8 * "[ $name ]\n"; color=:green)
    m = run_script_in_isolation("ex/ex_$name.jl")

    # include("./ex/ex_$name.jl")
end
