

def generic_denoiser(kernel_type: str, kernel_size: int):
    """
    Factory function that creates a descriptor function from a given configuration.

    Parameters
    ----------
    color_space : str
        Color space (e.g., "rgb", "hsv", "lab", "ycbcr", "gray").
    channels : list[str]
        Channel names to use.
    bins : list[int]
        Number of bins per channel.
    ranges : list[tuple[int,int]]
        Value ranges for each channel.
    weights : list[float]
        Weights to apply to each channel.

    Returns
    -------
    descriptor_fn : function
        Function that computes the concatenated histogram of the image.
    """

    def denoiser(img: NDArray) -> NDArray:
        if kernel_type == "gaussian":
            converted = img
            denoised = cv2.GaussianBlur(converted, (kernel_size, kernel_size), 0)
        elif kernel_type == "median":
            converted = img
            denoised = cv2.medianBlur(converted, kernel_size)
        elif kernel_type == "bilateral":
            converted = img
            denoised = cv2.bilateralFilter(converted, d=kernel_size, sigmaColor=75, sigmaSpace=75)
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")
        
        return denoised

    denoiser.__name__ = (
    f"{color_space}_{'_'.join(channels)}"
    f"_bins{'-'.join(map(str,bins))}"
    f"_w{'-'.join(map(str,weights))}"
    f"_hier{'-'.join(map(str,hierarchical_levels))}"
    )
    return denoiser
