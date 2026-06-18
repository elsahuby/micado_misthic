from astropy.io import fits 
import numpy as np  

def get_contrast_from_contrast_curve_file(fits_path, distance_in_mas):
    """
    Get the contrast value from a contrast curve FITS file at a given distance in milliarcseconds (mas).
    
    Parameters:
    - fits_path: str or Path, path to the FITS file containing the contrast curve.
    - distance_in_mas: float, the distance in milliarcseconds for which to retrieve the contrast.
    
    Returns:
    - contrast_value: float, the contrast value at the specified distance. Returns None if the distance is out of bounds.
    """
    
    # Open the FITS file and read the data
    with fits.open(fits_path) as hdul:
        data = hdul[1].data  # Assuming the contrast curve is in the first extension
    
    # Extract separation and contrast values
    separations = data['separation_mas']
    contrasts = data['contrast_5sigma']
    
    # Find the index of the closest separation value
    idx = (np.abs(separations - distance_in_mas)).argmin()
    
    # Check if the closest separation is within a reasonable range 
    if np.abs(separations[idx] - distance_in_mas) <= 1.0:
        return contrasts[idx]
    else:
        return None  # Return None if the distance is out of bounds