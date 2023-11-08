
translate_diff_id(G, χ, ξ, conv) = translate_diff(G, χ, identity_element(G), ξ, conv)

"""
The left translation ``T_L(g,ξ) = gξ``.
"""
move(G, χ, ξ, ::LeftSide) = translate_diff_id(G, χ, ξ, (LeftAction(), LeftSide()))
"""
The right translation ``T_R(g,ξ) = ξg``.
"""
move(G, χ, ξ, ::RightSide) = translate_diff_id(G, χ, ξ, (RightAction(), RightSide()))
