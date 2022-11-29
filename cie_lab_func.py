from utils_func import *

"""# Color Segmentation RGB to CIE LAB
### also applying Fire Candidate rules
"""

# implementing the rules


def isAllPixelBlack(frame):
    img = array_to_img(frame)
    extrema = img.convert("L").getextrema()
    return extrema == (0, 0)
    # if extrema == (0, 0):
    #     # all black
    # elif extrema == (1, 1):
    #     # all white


def LAB_rules(sub_lab):
    '''
    This method uses the CIE L*a*b* color space and performs 4 bitwise filters
    The method returns true at any pixel that satisfies:
    L* > Lm* (mean of L* values)
    a* > am* (mean of a* values)
    b* > bm* (mean of b* values)
    b* > a*
    '''
    L = sub_lab[:, :, 0]
    a = sub_lab[:, :, 1]
    b = sub_lab[:, :, 2]
    Lm = cv2.mean(L)
    am = cv2.mean(a)
    bm = cv2.mean(b)
    R1 = cv2.compare(L, Lm, cv2.CMP_GT)
    R2 = cv2.compare(a, am, cv2.CMP_GT)
    R3 = cv2.compare(b, bm, cv2.CMP_GT)
    R4 = cv2.compare(b, a, cv2.CMP_GT)
    R12 = cv2.bitwise_and(R1, R2)
    R34 = cv2.bitwise_and(R3, R4)
    R14 = cv2.bitwise_and(R1, R4)
    RALL = cv2.bitwise_and(R12, R34)

    L = RALL
    a = RALL
    b = RALL
    sub_lab[:, :, 0] = L
    sub_lab[:, :, 1] = a
    sub_lab[:, :, 2] = b
    return sub_lab, RALL


def applyRules(frame_original, rules):
    # TODO: apply filter
    filter = copy.deepcopy(frame_original)
    R = filter[:, :, 0]
    G = filter[:, :, 1]
    B = filter[:, :, 2]

    filter[:, :, 0] = cv2.bitwise_and(R, rules)
    filter[:, :, 1] = cv2.bitwise_and(G, rules)
    filter[:, :, 2] = cv2.bitwise_and(B, rules)
    return filter


def extract_single_dim_from_LAB_convert_to_RGB(image, idim):
    '''
    image is a single lab image of shape (None,None,3)
    '''
    z = np.zeros(image.shape)
    if idim != 0:
        z[:, :, 0] = 80  # I need brightness to plot the image along 1st or 2nd axis
    z[:, :, idim] = image[:, :, idim]
    z = lab2rgb(z)
    return(z)


def getLAB(Xsub_lab_rules, Nsample=1):
    labToSave = []
    count = 1
    fig = plt.figure(figsize=(13, 3*Nsample))
    size = 50

    axL = fig.add_subplot(Nsample, 3, count)
    lab_rgb_gray = extract_single_dim_from_LAB_convert_to_RGB(
        Xsub_lab_rules, 0)
    axL.imshow(lab_rgb_gray)
    axL.axis("off")
    axL.set_title("L: lightness")
    labToSave.append(255*lab_rgb_gray)
    count += 1

    axA = fig.add_subplot(Nsample, 3, count)
    lab_rgb_gray = extract_single_dim_from_LAB_convert_to_RGB(
        Xsub_lab_rules, 1)
    axA.imshow(lab_rgb_gray)
    axA.axis("off")
    axA.set_title("A: color spectrums green to red")
    labToSave.append(255*lab_rgb_gray)

    count += 1

    axB = fig.add_subplot(Nsample, 3, count)
    lab_rgb_gray = extract_single_dim_from_LAB_convert_to_RGB(
        Xsub_lab_rules, 2)
    axB.imshow(lab_rgb_gray)
    axB.axis("off")
    axB.set_title("B: color spectrums blue to yellow")
    labToSave.append(255*lab_rgb_gray)

    count += 1
    plt.show()
    return labToSave


def getCandidateFire(ROI):

    ROI01 = copy.deepcopy(ROI)
    # ROI01 = ROI01/255.0

    ROI_lab = rgb2lab(ROI01)

    # implementing rules
    ROI_lab_copy = copy.deepcopy(ROI_lab)
    ROI_lab_rules, applied_lab_rules = LAB_rules(ROI_lab_copy)
    #labToSave = getLAB(ROI_lab_rules)

    filteredRGBWithLABRules = applyRules(ROI, applied_lab_rules)
    return filteredRGBWithLABRules
