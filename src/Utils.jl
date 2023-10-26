
function assert_equal_actions(obj1, obj2, msg)
    a1 = get_action(obj1)
    a2 = get_action(obj2)
    if a1 != a2
        throw(ErrorException("$msg:\n\t$a1\n≠\n\t$a2"))
    end
end

algebra(G) = TangentSpace(G, identity_element(G))
