plain_manifold(M::ProductManifold) = ProductManifold(plain_manifold.(M.manifolds)...)
plain_manifold(M) = base_manifold(M)

function assert_equal_properties(obj1, prop1, obj2, prop2, msg)
    a1 = prop1(obj1)
    a2 = prop2(obj2)
    if a1 != a2
        throw(ErrorException("$msg:\n\t$a1\n≠\n\t$a2"))
    end
end

