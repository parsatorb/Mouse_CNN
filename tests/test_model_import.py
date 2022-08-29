import mousenet

def test_generic_import():
    model = mousenet.load(pretraining=None)

def test_kaiming_import():
    model = mousenet.load(pretraining="kaiming")
