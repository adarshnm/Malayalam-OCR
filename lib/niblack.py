import numpy as np
''' Converted from matlab code made by Jan Motl

Jan Motl (2020). Niblack local thresholding 
(https://www.mathworks.com/matlabcentral/fileexchange/40849-niblack-local-thresholding), 
MATLAB Central File Exchange. Retrieved February 14, 2020.

  '''


def average_filter(image, window_size=[3, 3], padding='constant'):
    dt = image.dtype
    m = window_size[0]
    n = window_size[1]

    if m % 2 == 0:
        m = m-1
    if n % 2 == 0:
        n = n-1

    if image.ndim != 2:
        print("Input image must be grayscale")
        return

    (rows, columns) = image.shape

    # pad image
    img_pre_pad = np.pad(image,
                         pad_width=((int((m+1)/2), 0), (int((n+1)/2), 0)), mode=padding)
    img_post_pad = np.pad(img_pre_pad,
                          pad_width=((0, int((m-1)/2)), (0, int((n+1)/2))), mode=padding)

    # converting to float
    imageF = img_post_pad.astype('float64')
    # Matrix t = is sum of numbers on left and above current cell
    t = np.cumsum(np.cumsum(imageF, axis=0), axis=1)

    # Calculate the mean value from look up table  `t`
    imageI = t[m:rows+m, n:columns+n] + t[0:rows, 0:columns] - \
        t[m:rows+m, 0:columns] - t[0:rows, n:columns+n]

    # Now each pixel contains sum of the window . But we want average value
    imageI = imageI/(m*n)

    img_return = np.array(imageI, dtype=dt)
    return img_return


def binarize(image, window_size=[3, 3], k=-0.2, offset=0, padding='constant'):
    ''' Niblack local thresholding binarization 

     Example
     -------
     import niblack

     img = cv2.imread("test.png", cv2.IMREAD_GRAYSCALE)
     bin_img = niblack.binarize(image=img,size=[25,25],offset=10,padding='symmetric')

    '''

    if image.ndim != 2:
        print("Input image must be grayscale")
        return

    # Convert to float
    image = image.astype('float64')

    # Mean value
    mean = average_filter(image, window_size=window_size, padding=padding)

    # Standard deviation
    meanSquare = average_filter(
        image**2, window_size=window_size, padding=padding)
    deviation = (meanSquare - mean**2)**0.5

    output = np.zeros(image.shape)

    output[image > mean + k * deviation - offset] = 1

    return output
