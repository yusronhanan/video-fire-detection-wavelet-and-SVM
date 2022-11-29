from utils_func import *

historyGMM = 5
varTGMM = 0
fgbgGMM = cv2.createBackgroundSubtractorMOG2(
    history=historyGMM, varThreshold=varTGMM, detectShadows=False)

"""# GMM Background Subtraction"""


def gmm_background_subtraction(frame, fgbgGMM):
    # frameHSV = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    mask = fgbgGMM.apply(frame)
    kernel = np.ones((6, 6), np.uint8)
    # kernel_dilation = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_opening = np.ones((2, 2), np.uint8)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_opening)
    # closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    # dilate = cv2.dilate(closing, kernel_dilation, iterations=3)
    colorObj = cv2.bitwise_and(frame, frame, mask=opening)
    mask = opening
    return mask, colorObj


class RectangleAxis:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h


def getObjectList(frame, colorObj, MIN_SIZE_FOR_MOVEMENT=1000):
    # apply bounding box
    colorObj_copy = copy.deepcopy(colorObj)
    frame_copy = copy.deepcopy(frame)
    gray = cv2.cvtColor(colorObj, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)[1]
    # Fill in holes via dilate(), and find contours of the thesholds
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(
        thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    # loop over the contours
    ROI_number = 0
    ROI_list = []
    ROI_rectangle_list = []
    for c in cnts:

        # Save the coordinates of all found contours
        (x, y, w, h) = cv2.boundingRect(c)

        # If the contour is too small, ignore it, otherwise, there's transient
        # movement
        if cv2.contourArea(c) > MIN_SIZE_FOR_MOVEMENT:
            # Draw a rectangle around big enough movements
            # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            rect = RectangleAxis(x, y, w, h)
            ROI_rectangle_list.append(rect)
            ROI = colorObj_copy[y:y+h, x:x+w]
            ROI_list.append(ROI)
            ROI_number += 1
    return ROI_list, ROI_rectangle_list, cnts


def drawRectangle(frame, rectAxis, text=""):
    x = rectAxis.x
    y = rectAxis.y
    w = rectAxis.w
    h = rectAxis.h
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if text != "":
        cv2.putText(frame, text, (x+w+10, y+h), 0, 0.3, (0, 255, 0))


def drawEdge(frame, contours):
    cv2.drawContours(image=frame, contours=contours, contourIdx=-1,
                     color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)


def fillBlack(frame, ROI_rectangle_list):
    black_px = np.asarray([0, 0, 0])
    for i in range(len(ROI_rectangle_list)):
        rectAxis = ROI_rectangle_list[i]
        x = rectAxis.x
        y = rectAxis.y
        w = rectAxis.w
        h = rectAxis.h
        frame[y:y+h, x:x+w] = black_px
