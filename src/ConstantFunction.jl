"""
    ConstantFunction(v)

A constant function always taking the value `v`.

The functions value can be accessed by providing either zero
or one (arbitrary) argument:

```jldoctest
f = ConstantFunction(10)

f(x)
f()

# output
10
```
"""
struct ConstantFunction{T} <: Function
    value::T
end

(c::ConstantFunction)(::Any) = c()
(c::ConstantFunction)() = c.value
Base.:*(s, c::ConstantFunction) = ConstantFunction(s*c.value)

Base.show(io::IO, f::ConstantFunction) = print(io, "$(f.value)")
