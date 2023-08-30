def crop(preprocessed_img, width, height, x0, z0):
    #slice from image
    # width = 256 * 4
    # height = 256 * 2
    # x0 = 135
    # z0= 350
    #check crop out of image
    h,w = preprocessed_img.shape
    if z0>h or z0+height> h or x0>w or x0+width > w:
        print("Did not crop, image is too small.")
        return preprocessed_img
    return preprocessed_img[z0:z0+height, x0:x0+width]
