def crop(preprocessed_img, width, height, x0, z0):
    #slice from image
    # width = 256 * 4
    # height = 256 * 2
    # x0 = 135
    # z0= 350
    return preprocessed_img[z0:z0+height, x0:x0+width]
