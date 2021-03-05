import cv2
import base64

def img_resize(image, max_pixel):
    height, width = image.shape[0], image.shape[1]
    if height > width and height > max_pixel:
        width = int(width / height * max_pixel)
        height = max_pixel
        img_new = cv2.resize(image, (width, height))
        return img_new
    elif height < width and width > max_pixel:
        height = int(height / width * max_pixel)
        width = max_pixel
        img_new = cv2.resize(image, (width, height))
        return img_new
    return image

def scaled_base64(img, max_pixel):
    im = cv2.imread(img)
    if max_pixel is not None:
        im = img_resize(im, max_pixel)
    image = cv2.imencode('.jpg', im)[1]
    base64_data = base64.b64encode(image)
    base64_data = bytes.decode(base64_data)
    return base64_data
