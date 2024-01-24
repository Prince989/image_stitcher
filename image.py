import numpy as np
import cv2 
import glob 
import imutils

cv2.ocl.setUseOpenCL(False)

def post_process(image):
	# check to see if we supposed to crop out the largest rectangular
	# region from the stitched image
    # create a 10 pixel border surrounding the stitched image
    print("[INFO] cropping...")
    stitched = cv2.copyMakeBorder(image, 10, 10, 10, 10,
        cv2.BORDER_CONSTANT, (0, 0, 0))
    # convert the stitched image to grayscale and threshold it
    # such that all pixels greater than zero are set to 255
    # (foreground) while all others remain 0 (background)
    gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
    # find all external contours in the threshold image then find
    # the *largest* contour which will be the contour/outline of
    # the stitched image
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    # allocate memory for the mask which will contain the
    # rectangular bounding box of the stitched image region
    mask = np.zeros(thresh.shape, dtype="uint8")
    (x, y, w, h) = cv2.boundingRect(c)
    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
    # create two copies of the mask: one to serve as our actual
    # minimum rectangular region and another to serve as a counter
    # for how many pixels need to be removed to form the minimum
    # rectangular region
    minRect = mask.copy()
    sub = mask.copy()
    # keep looping until there are no non-zero pixels left in the
    # subtracted image
    while cv2.countNonZero(sub) > 0:
        # erode the minimum rectangular mask and then subtract
        # the thresholded image from the minimum rectangular mask
        # so we can count if there are any non-zero pixels left
        minRect = cv2.erode(minRect, None)
        sub = cv2.subtract(minRect, thresh)

    # find contours in the minimum rectangular mask and then
    # extract the bounding box (x, y)-coordinates
    cnts = cv2.findContours(minRect.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea,default=0)
    (x, y, w, h) = cv2.boundingRect(c)
    # use the bounding box coordinates to extract the our final
    # stitched image
    stitched = stitched[y:y + h, x:x + w]

    return stitched


image_paths = glob.glob('photos5/*.jpg')

images = []
images2 = []

for image in image_paths:
    img = cv2.imread(image)
    images.append(img)


# for image in image_paths[27:]:
#     img = cv2.imread(image)
#     images2.append(img)

imageStitcher = cv2.Stitcher_create()

error , stitchedImage = imageStitcher.stitch(images)

# cv2.imwrite('./output1.png',stitchedImage)

# error2 , stitchedImage2 = imageStitcher.stitch(images2)

# cv2.imwrite('./output2.png',stitchedImage2)

# images3 = []

# images3.append(stitchedImage)
# images3.append(stitchedImage2)

# error3 , stitchedImage3 = imageStitcher.stitch(images3)

if not error:
    cv2.imwrite('./output.png',stitchedImage)
    # cv2.imshow('Image', stitchedImage)
    # cv2.waitKey(0)
else:
    # print(error,error2,error3)
    print(error)
