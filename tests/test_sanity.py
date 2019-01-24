from draugr import terminal_plot
import numpy as np

def test_sanity():
    assert True
    assert False is not True
    answer_to_everything = str(42)
    assert str(42) == answer_to_everything

def test_plot():
  terminal_plot(np.tile(range(9), 4), plot_character='o')
  assert True

if __name__ == '__main__':
    test_plot()