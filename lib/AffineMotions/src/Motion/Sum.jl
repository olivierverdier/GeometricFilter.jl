
struct AffineMotionSum{TA,TV<:Tuple} <: CompositeAffineMotion{TA}
    motions::TV
end

AffineMotionSum(motions::SimpleAffineMotion{TA}...) where {TA} = AffineMotionSum{TA, typeof(motions)}(motions)

Base.show(io::IO, ms::AffineMotionSum) = print(io, "AffineMotionSum(" * join([repr(m) for m in ms.motions], ", ") * ")")

get_action(m::AffineMotionSum) = get_action(first(m.motions))

get_dynamics(m::AffineMotionSum, x) = sum([get_dynamics(m_, x) for m_ in m.motions])

get_lin(m::AffineMotionSum) = ξ -> sum([get_lin(m_)(ξ) for m_ in m.motions])

rescale_motion(s::Number, sm::AffineMotionSum) = AffineMotionSum([rescale_motion(s, m) for m in sm.motions]...)

Base.:+(motions::SimpleAffineMotion{TA}...) where {TA} = AffineMotionSum(motions...)

Base.:+(sms::AffineMotionSum{TA}...) where {TA} = AffineMotionSum(reduce(vcat, [collect(sm.motions) for sm in sms])...)

Base.:+(sms::AffineMotionSum{TA}, m::SimpleAffineMotion) where {TA} = sms + AffineMotionSum(m)
Base.:+(m::SimpleAffineMotion, sms::AffineMotionSum{TA}) where {TA} = AffineMotionSum(m) + sms

# Tricky here, this `isapprox` is too rigid.
Base.isapprox(M1::AffineMotionSum, M2::AffineMotionSum; kwargs...) = all(isapprox(m1, m2; kwargs...) for (m1,m2) in zip(M1.motions, M2.motions))

swap_group_motion(ms::AffineMotionSum{TA}) where {TA<:Union{GroupOperationAction{LeftAction, TAS} where {TAS}}} = AffineMotionSum([swap_group_motion(m) for m in ms.motions]...)
