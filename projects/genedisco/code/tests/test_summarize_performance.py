import math

from reproduction.summarize_performance import mean_std_sem


def test_mean_std_sem_matches_paper_population_sem():
    mean, std, sem = mean_std_sem([0.0, 0.2])

    assert mean == 0.1
    assert std == 0.1
    assert sem == 0.1 / math.sqrt(2)
