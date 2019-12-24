import numpy as np

import vaex


def test_percentile_approx():
    df = vaex.example()
    # Simple test
    percentile = df.percentile_approx('z', percentage=99)
    expected_result = 15.1739
    np.testing.assert_almost_equal(percentile, expected_result, decimal=3)

    # Test for multiple percentages
    percentiles = df.percentile_approx('x', percentage=[25, 50, 75], percentile_shape=65536)
    expected_result = [-3.5992, -0.0367, 3.4684]
    np.testing.assert_array_almost_equal(percentiles, expected_result, decimal=3)

    # Test for multiple expressions
    percentiles_2d = df.percentile_approx(['x', 'y'], percentage=[33, 66])
    expected_result = np.array(([-2.3310, 1.9540], [-2.4313, 2.1021]))
    np.testing.assert_array_almost_equal(percentiles_2d, expected_result, decimal=3)
