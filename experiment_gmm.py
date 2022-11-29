from gmm_func import *


def main_gmm(input_loc, output_loc, fgbgGMM):
    """Function to extract frames from input video file
    and save them as separate frames in an output directory.
    Args:
        input_loc: Input video file.
        output_loc: Output directory to save the frames.
    Returns:
        None
    """
    # Log the time
    time_start = time.time()
    # Start capturing the feed
    cap = cv2.VideoCapture(input_loc)
    # Find the number of frames
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    print("Number of frames: ", video_length)
    count = 0
    # number of frames to skip
    numFrameToSave = 10
    print("Converting video..\n")
    # Start converting the video
    ROI_original_list_loc = []

    while cap.isOpened():
        # Extract the frame
        ret, frame = cap.read()
        if not ret:
            break
        # get RGB format
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if (count % numFrameToSave == 0):
            saveOneFrame(output_loc, frame, "Original - "+str(count+1),
                         isSave=False, isRGB2BGR=True)

            # get ROI
            if count != 0:
                # GMM
                mask, colorObj = gmm_background_subtraction(frame, fgbgGMM)
                saveOneFrame(output_loc, mask, "Mask - "+str(count+1),
                             isSave=True, isRGB2BGR=True)
                mask_copy = copy.deepcopy(mask)
                maskS = cv2.resize(cv2.cvtColor(
                    mask_copy, cv2.COLOR_RGB2BGR), (960, 540))
                cv2.imshow('Mask', maskS)

                ROI_list, ROI_rectangle_list, contours_ROI_list = getObjectList(
                    frame, colorObj, MIN_SIZE_FOR_MOVEMENT=1000)
                roi_original_toSave = []
                for i in range(len(ROI_list)):
                    r = ROI_list[i]
                    rectAxis = ROI_rectangle_list[i]
                    drawRectangle(frame, rectAxis)
                    roi_original_toSave.append(r)

                ROI_originals_loc = saveFrame(output_loc, roi_original_toSave,
                                              str(count+1) + " - ROI", isSave=True, isRGB2BGR=True)
                ROI_original_list_loc.extend(ROI_originals_loc)
            # else:
            #     fgbgGMM = cv2.createBackgroundSubtractorMOG2(
            #         history=historyGMM, varThreshold=varTGMM, detectShadows=False)
                # Save the frame in list
                saveOneFrame(output_loc, frame, "Original - "+str(count+1)+" - BB ",
                             isSave=True, isRGB2BGR=True)
                saveOneFrame(output_loc, colorObj, "Color Object - "+str(count+1),
                             isSave=True, isRGB2BGR=True)

                frame_copy = copy.deepcopy(frame)
                colorObj_copy = copy.deepcopy(colorObj)

                frameS = cv2.resize(cv2.cvtColor(
                    frame_copy, cv2.COLOR_RGB2BGR), (960, 540))

                colorObjS = cv2.resize(cv2.cvtColor(
                    colorObj_copy, cv2.COLOR_RGB2BGR), (960, 540))

                cv2.imshow('Frame', frameS)
                cv2.imshow('Color Obj', colorObjS)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
        count = count + 1
        # If there are no more frames left
        if (count > (video_length-1)):
            break
    # Log the time again
    time_end = time.time()
    # Release the feed
    cap.release()
    # Print stats
    print("Done extracting frames.\n%d frames extracted" % count)
    print("Only produce for every %d frame" % numFrameToSave)
    print("It took %d seconds forconversion." % (time_end-time_start))


def run_video_by_dir_gmm(directory, isFire, trial, dev="Training"):
    listVideo = list(os.listdir(directory))
    isFireToString = 'Fire' if isFire else 'Non Fire'
    # change limit, limit > 0, no limit otherwise
    limit = -1
    N = len(listVideo)
    if limit > 0:
        N = limit
    for i in range(N):
        # print(listVideo)
        # for video in listVideo:
        video = listVideo[i]
        if 'not' in video:
            continue
        print(video+' start')
        fileName, ext = os.path.splitext(video)
        input_loc = directory+fileName+ext
        output_loc = '../content/drive/My Drive/A TA/' + \
            dev+'/'+trial+'/'+isFireToString+'/'+fileName
        try:
            os.makedirs(output_loc)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
                # time.sleep might help here
            pass
        main_gmm(
            input_loc, output_loc, fgbgGMM)
        print(video+' end')

# Training


def main_experiment(directories, trial, dev):
    time_start = time.time()
    for d in directories:
        print(dev)
        print(d[0])
        run_video_by_dir_gmm(d[0], d[1], trial, dev=dev)
    time_end = time.time()
    print("It took %d seconds in total." % (time_end-time_start))


# dir_training = [
#     ['../content/drive/My Drive/DATA SET/Training 2/Fire/', True],
#     ['../content/drive/My Drive/DATA SET/Training 2/Non Fire/', False],
# ]
trial = '--final-1212-0'
# main_experiment(dir_training, trial, "ROI from GMM Training")

dir_testing = [
    ['../content/drive/My Drive/DATA SET/Testing 2/Fire/', True],
    # ['../content/drive/My Drive/DATA SET/Testing/Non Fire/', False],
]
main_experiment(dir_testing, trial, "ROI from GMM Testing")
