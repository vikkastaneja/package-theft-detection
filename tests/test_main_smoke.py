# tests/test_main_smoke.py
def test_main_importable():
    # Just ensure main module can be imported without side effects/crash.
    import main  # noqa: F401