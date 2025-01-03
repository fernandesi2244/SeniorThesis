import numpy as np

def load_volume_components(filename):
    """
    Read in 3-D magnetic field components from generated C code output.
    """
    
    # Load binary data from file
    with open(filename, "rb") as f:
        data = np.fromfile(f, dtype=np.float64)  # Assuming double precision float
    
    nx, ny, nz = 200, 400, 100
    
    flattened_size = nx * ny * nz

    # Initialize 3D magnetic field arrays
    bx_3D = data[:flattened_size].reshape((nx, ny, nz))
    by_3D = data[flattened_size:2*flattened_size].reshape((nx, ny, nz))
    bz_3D = data[2*flattened_size:3*flattened_size].reshape((nx, ny, nz))

    return bx_3D, by_3D, bz_3D