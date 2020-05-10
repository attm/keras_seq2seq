from keras_generic_seq2seq import *


def test_build_model():
    m, e, d = build_model(2000)
    assert m != None
    assert e != None
    assert d != None

    m, e, d = build_model(2000, gru=True, additive_attention=False)
    assert m != None
    assert e != None
    assert d != None

    m, e, d = build_model(2000, gru=False, additive_attention=False)
    assert m != None
    assert e != None
    assert d != None

    m, e, d = build_model(2000, gru=False, additive_attention=True)
    assert m != None
    assert e != None
    assert d != None