import pytest
from anpr.utils.utils import read_license_plate, get_car

# def test_read_license_plate():
#     """
#     Test the read_license_plate function.
#     """
#     # Add a sample test image or mock image for testing
#     # This is a placeholder test
#     assert read_license_plate(None) == (None, None)

# def test_get_car():
#     """
#     Test the get_car function.
#     """
#     # Placeholder test with mock data
#     license_plate = (10, 20, 30, 40, 0.9, 1)
#     vehicle_track_ids = [(5, 10, 35, 45, 1)]
    
#     result = get_car(license_plate, vehicle_track_ids)
#     assert result == (5, 10, 35, 45, 1)

def test_get_car_no_match():
    """
    Test get_car function when no matching vehicle is found.
    """
    license_plate = (100, 200, 300, 400, 0.9, 1)
    vehicle_track_ids = [(5, 10, 35, 45, 1)]
    
    result = get_car(license_plate, vehicle_track_ids)
    assert result == (-1, -1, -1, -1, -1)
