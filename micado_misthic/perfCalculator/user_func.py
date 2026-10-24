"""
Author: Tristan Deseine
"""

from astropy.io import fits 
import numpy as np  

def get_contrast_from_contrast_curve_file(fits_path, distance_in_mas):
    """
    Get the contrast value from a contrast curve FITS file at a given distance in milliarcseconds (mas).
    
    Parameters:
    - fits_path: str or Path, path to the FITS file containing the contrast curve.
    - distance_in_mas: float, the distance in milliarcseconds for which to retrieve the contrast.
    
    Returns:
    - contrast_value: float, the nearest sampled contrast value.
      Returns None if the distance is outside the sampled separation range.
    """
    
    # Open the FITS file and read the data
    data = fits.getdata(fits_path) # the contrast curve is in the second row
    
    # Extract separation and contrast values
    separations = data[0]
    contrasts = data[1]
    
    distance_in_mas = float(distance_in_mas)
    if not np.isfinite(distance_in_mas):
        raise ValueError("Distance must be finite.")
    valid = np.isfinite(separations)
    separations = separations[valid]
    contrasts = contrasts[valid]
    if separations.size == 0:
        raise ValueError("The contrast curve contains no valid separations.")
    if not separations.min() <= distance_in_mas <= separations.max():
        return None

    # Use the nearest sample inside the curve, regardless of grid spacing.
    idx = (np.abs(separations - distance_in_mas)).argmin()
    return float(contrasts[idx])
