from common.paths import PROJECT_ROOT, DATA_ROOT

def test_expected_layout():
    assert (PROJECT_ROOT / 'question1').is_dir()
    assert (PROJECT_ROOT / 'question2').is_dir()
    assert (PROJECT_ROOT / 'question3').is_dir()
    assert DATA_ROOT.name == 'data' or DATA_ROOT.is_absolute()
