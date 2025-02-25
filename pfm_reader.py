import numpy as np

def read_pfm(file):
    """Reads a PFM file and returns an image as a NumPy array."""
    header = file.readline().decode('utf-8').strip()
    color = header == 'PF'

    dim_line = file.readline().decode('utf-8')
    width, height = map(int, dim_line.split())

    scale = float(file.readline().decode('utf-8'))
    endian = '<' if scale < 0 else '>'
    
    # Read remaining bytes
    data = file.read()
    data = np.frombuffer(data, endian + 'f')  # ✅ Use frombuffer() instead of fromfile()

    shape = (height, width, 3) if color else (height, width)

    pfm_data = np.flipud(data.reshape(shape))

    # Normalize and convert to uint8
    pfm_data = (pfm_data - pfm_data.min()) / (pfm_data.max() - pfm_data.min()) * 255
    pfm_data = pfm_data.astype(np.uint8)

    # Convert grayscale to RGB if needed
    if not color:
        pfm_data = np.stack([pfm_data] * 3, axis=-1)

    return pfm_data
