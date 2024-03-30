import subprocess


def test_main():
    assert subprocess.check_output(["inter-active-learning", "foo", "foobar"], text=True) == "foobar\n"
