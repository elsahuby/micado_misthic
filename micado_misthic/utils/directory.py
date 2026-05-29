import os

def create_output_directory(output_directory):
    """
    This functions aimes to create the 3 CLC directories in the output directory, if they do not already exist.
    Then, in each CLC directory, it creates 2 subdirectories for the 2 quasi-static speckle cases (NCPA and NoNCPA).
    Then, in each of these subdirectories, it creates 3 subdirectories for the 3 quartiles (Q1, MED, Q4).
    Then, in each of these subdirectories, it creates 2 subdirectories for the 2 sampling cases (samp1.5 and samp4.0).
    Finally, there will be the different filter directories in which the different simulation outputs will be stored.
    Parameters
    ----------
    output_directory : str
        Path to the output directory where the CLC directories will be created

    """
    CLC= ["CLC0", "CLC1", "CLC2"]
    quartile= ["Q1", "MED",  "Q4"]
    quasi_static_speckle = ["NoNCPA", "NCPA"]
    sampling = ["samp1.5", "samp4.0"]
    filter = ["filter1.190", "filter1.245", "filter1.270", "filter1.582","filter1.635" ,"filter1.693", "filter2.100", "filter2.145", "filter2.235"]
    for c in CLC:
        for n in quasi_static_speckle:
            for q in quartile:
                    for f in filter:
                        for s in sampling:
                            path = os.path.join(output_directory, c, n, q, f, s)
                            os.makedirs(path, exist_ok=True)

    return 

if __name__ == "__main__":
    output_directory = r'C:\Users\tdeseine\Desktop\MISTHIC\output\opti_tristan'
    create_output_directory(output_directory)
                        
    
