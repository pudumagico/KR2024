#bias("
is_attr_value(ID, X) :- query(TO, TI, ATTR), state(TI, ID), has_attr(ID, ATTR, X).
is_attr(X) :- query(TO, TI, ATTR), state(TI, ID), has_attr(ID, ATTR, X).
state(TO,ID) :- scene(TO), object(ID).
ans(V) :- end(TO), attr_value(TO,V).
ans(V) :- end(TO), attr(TO,V).
ans(V) :- end(TO), rel(TO,V).
ans(V) :- end(TO), bool(TO,V).
:- not ans(_).
state(TO,ID) :- select(TO, TI, CLASS), state(TI, ID), has_attr(ID, class, CLASS).
state(TO,ID) :- filter(TO, TI, ATTR, VALUE), state(TI, ID), has_attr(ID, ATTR, VALUE).
state(TO,ID) :- filter_any(TO, TI, VALUE), state(TI, ID), has_attr(ID, ATTR, VALUE).
state(TO, ID2) :- relate(TO, TI, CLASS, REL, subject), state(TI, ID), has_attr(ID2, class, CLASS), has_rel(ID2, REL, ID).
state(TO, ID2) :- relate(TO, TI, CLASS, REL, object), state(TI, ID), has_attr(ID2, class, CLASS), has_rel(ID, REL, ID2).
state(TO, ID2) :- relate_any(TO, TI, REL, subject), state(TI, ID), has_rel(ID2, REL, ID).
state(TO, ID2) :- relate_any(TO, TI, REL, object), state(TI, ID), has_rel(ID, REL, ID2).
state(TO, ID2) :- relate_attr(TO, TI, CLASS, ATTR), state(TI, ID), has_attr(ID, ATTR, VALUE), has_attr(ID2, class, CLASS), has_attr(ID2, ATTR, VALUE2), VALUE==VALUE2, ID!=ID2.
attr_value(TO,VALUE) :- query(TO, TI, ATTR), state(TI, ID), has_attr(ID, ATTR, VALUE).
bool(TO, yes) :- verify_attr(TO, TI, ATTR, VALUE), state(TI, ID), has_attr(ID, ATTR, VALUE).
bool(TO,no) :- verify_attr(TO, TI, ATTR, VALUE), not bool(TO,yes).
bool(TO, yes) :- verify_rel(TO, TI, CLASS, REL, subject), state(TI, ID), has_attr(ID2, class, CLASS), has_rel(ID2, REL, ID).
bool(TO,no) :- verify_rel(TO, TI, CLASS, REL, subject), not bool(TO,yes).
bool(TO, yes) :- verify_rel(TO, TI, CLASS, REL, object), state(TI, ID), has_attr(ID2, class, CLASS), has_rel(ID, REL, ID2).
bool(TO,no) :- verify_rel(TO, TI, CLASS, REL, object), not bool(TO,yes).
attr_value(TO, VALUE) :- choose_attr(TO, TI, ATTR, VALUE, VALUE2), state(TI, ID), has_attr(ID, ATTR, VALUE).
attr_value(TO, VALUE2) :- choose_attr(TO, TI, ATTR, VALUE, VALUE2), state(TI, ID), has_attr(ID, ATTR, VALUE2).
rel(TO, REL) :- choose_rel(TO, TI, CLASS, REL, REL2, subject), state(TI, ID), has_attr(ID2, class, CLASS), has_rel(ID2, REL, ID).
rel(TO, REL2) :- choose_rel(TO, TI, CLASS, REL, REL2, subject), state(TI, ID), has_attr(ID2, class, CLASS), has_rel(ID2, REL2, ID).
rel(TO, REL) :- choose_rel(TO, TI, CLASS, REL, REL2, object), state(TI, ID), has_attr(ID2, class, CLASS), has_rel(ID, REL, ID2).
rel(TO, REL2) :- choose_rel(TO, TI, CLASS, REL, REL2, object), state(TI, ID), has_attr(ID2, class, CLASS), has_rel(ID, REL2, ID2).
bool(TO,no) :- all_different(TO, TI, ATTR), state(TI, ID), state(TI, ID2), has_attr(ID, ATTR, VALUE), has_attr(ID2, ATTR, VALUE).
bool(TO,yes) :- all_different(TO, TI, ATTR), not bool(TO,no).
bool(TO,no) :- all_same(TO, TI, ATTR), state(TI, ID), state(TI, ID2), has_attr(ID, ATTR, VALUE), not has_attr(ID2, ATTR, VALUE).
bool(TO,yes) :- all_same(TO, TI, ATTR), not bool(TO,no).
bool(TO, yes) :- two_different(TO, TI0, TI1, ATTR), state(TI0, ID), state(TI1, ID2), has_attr(ID, ATTR, VALUE), has_attr(ID2, ATTR, VALUE2), VALUE != VALUE2.
bool(TO, yes) :- two_different(TO, TI0, TI1, ATTR), state(TI0, ID), state(TI1, ID2), has_attr(ID, ATTR, _), not has_attr(ID2, ATTR, _).
bool(TO, yes) :- two_different(TO, TI0, TI1, ATTR), state(TI0, ID), state(TI1, ID2), not has_attr(ID, ATTR, _), has_attr(ID2, ATTR, _).
bool(TO,no) :- two_different(TO, TI0, TI1, ATTR), not bool(TO,yes).
bool(TO, yes) :- two_same(TO, TI0, TI1, ATTR), state(TI0, ID), state(TI1, ID2), has_attr(ID, ATTR, VALUE), has_attr(ID2, ATTR, VALUE2), VALUE == VALUE2.
bool(TO,no) :- two_same(TO, TI0, TI1, ATTR), not bool(TO,yes).
attr(TO, ATTR) :- common(TO, TI0, TI1), state(TI0, ID), state(TI1, ID2), has_attr(ID, ATTR, VALUE), has_attr(ID2, ATTR, VALUE), ATTR != name, ATTR != class, ATTR != hposition, ATTR != vposition.
state(TO,ID) :- compare(TO, TI0, TI1, VALUE, true), state(TI0, ID), state(TI1, ID2), has_attr(ID, _, VALUE), not has_attr(ID2, _, VALUE).
state(TO,ID2) :- compare(TO, TI0, TI1, VALUE, true), state(TI0, ID), state(TI1, ID2), not has_attr(ID, _, VALUE), has_attr(ID2, _, VALUE).
state(TO,ID2) :- compare(TO, TI0, TI1, VALUE, false), state(TI0, ID), state(TI1, ID2), has_attr(ID, _, VALUE), not has_attr(ID2, _, VALUE).
state(TO,ID) :- compare(TO, TI0, TI1, VALUE, false), state(TI0, ID), state(TI1, ID2), not has_attr(ID, _, VALUE), has_attr(ID2, _, VALUE).
bool(TO,yes) :- and(TO, TI0, TI1), bool(TI0,yes), bool(TI1,yes).
bool(TO,no) :- and(TO, TI0, TI1), not bool(TO,yes).
bool(TO,yes) :- or(TO, TI0, TI1), bool(TI0,yes).
bool(TO,yes) :- or(TO, TI0, TI1), bool(TI1,yes).
bool(TO,no) :- or(TO, TI0, TI1), not bool(TO,yes).
state(TO, ID) :- negate(TO, TI0, TI1), state(TI1, ID), not state(TI0, ID).

{state(TO,ID): state(TI,ID)} = 1 :- unique(TO, TI).
{attr(TO, ATTR): is_attr(ATTR)} = 1 :- common(TO, TI0, TI1).
{ has_attr(ID, ATTR, VALUE) : is_attr_value(ID, VALUE)} = 1 :- query(TO, TI, ATTR), state(TI, ID), ATTR != name, ATTR != class, ATTR != hposition, ATTR != vposition.
{has_attr(ID, ATTR, VALUE); has_attr(ID, ATTR, VALUE2)} = 1 :- choose_attr(TO, TI, ATTR, VALUE, VALUE2), state(TI, ID).
{has_rel(ID2, REL, ID): has_attr(ID2, class, CLASS); has_rel(ID2, REL2, ID): has_attr(ID2, class, CLASS)} = 1 :- choose_rel(TO, TI, CLASS, REL, REL2, subject), state(TI, ID).
{has_rel(ID, REL, ID2): has_attr(ID2, class, CLASS); has_rel(ID, REL2, ID2): has_attr(ID2, class, CLASS)} = 1 :- choose_rel(TO, TI, CLASS, REL, REL2, object), state(TI, ID).
:~ unique(TO, TI), state(TO,ID), has_obj_weight(ID, P). [P, (TO, ID)]
").