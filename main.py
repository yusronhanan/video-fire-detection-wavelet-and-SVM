from numpy import NaN
from utils_func import *
from gmm_func import *
from cie_lab_func import *
from wavelet_func import *
from classifier_func import *
from env_main import *


# df_test = pd.read_csv(file_input_generated_df +
#                       '-Frame Testing -- with label.csv')
df_test = joblib.load(out_df+trial+'-'+dir_testing_with_label+'.pkl')
empty_label_pred_list = [NaN for i in range(len(df_test))]
df_test["label_pred"] = empty_label_pred_list


classifier = joblib.load(file_input_generated_df+'-' +
                         clf_filename+'-' + svm_kernel+'.pkl')


# roi
df_test_roi = joblib.load(out_df+trial+'-'+roi_testing_with_label+'.pkl')
empty_label_pred_list_roi = [NaN for i in range(len(df_test_roi))]
empty_label_EB_list_roi = [NaN for i in range(len(df_test_roi))]

df_test_roi["label_pred"] = empty_label_pred_list_roi
df_test_roi["energy_block"] = empty_label_EB_list_roi


def main(filename_dir, input_loc, output_loc, fgbgGMM, isFireDirString):
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
    vwidth = 480
    vheight = 270
    # number of frames to skip
    numFrameToSave = 10
    print("Converting video..\n")
    # Start converting the video
    isFire_list_pred_by_frame = []
    EBlock_list_pred_by_ROI = []
    ROI_fileName_list = []
    isFire_list_pred_by_ROI = []
    video_list = []
    n_frame_list = []
    while cap.isOpened():
        # Extract the frame
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (vwidth, vheight))
        if (count % numFrameToSave == 0):
            # get RGB format
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            saveOneFrame(output_loc, frame, "Original - "+str(count+1),
                         isSave=False, isRGB2BGR=True)
            frame_original = copy.deepcopy(frame)
            # GMM

            # get ROI
            isFireInFrame = 0
            if count != 0:
                mask, colorObj = gmm_background_subtraction(frame, fgbgGMM)
                saveOneFrame(output_loc, mask, "Mask - "+str(count+1),
                             isSave=True, isRGB2BGR=True)
                colorObj_with_contours = copy.deepcopy(colorObj)

                colorObj_WithLABRules = getCandidateFire(colorObj)
                # kernel = np.ones((1, 1), np.uint8)
                # colorObj_WithLABRules = cv2.morphologyEx(
                #     colorObj_WithLABRules, cv2.MORPH_OPEN, kernel)
                ROI_list, ROI_rectangle_list, contours_ROI_list = getObjectList(
                    frame, colorObj_WithLABRules, MIN_SIZE_FOR_MOVEMENT=100)
                # roi_original_toSave = []
                # roi_not_actual_fire_toSave = []
                # roi_actual_fire_toSave = []
                for i in range(len(ROI_list)):
                    r = ROI_list[i]
                    roi_filename = str(count+1) + " - ROI " + str(i+1)
                    rectAxis = ROI_rectangle_list[i]
                    r_rgb = frame[rectAxis.y:rectAxis.y +
                                  rectAxis.h, rectAxis.x:rectAxis.x+rectAxis.w]
                    r_colorObj = colorObj[rectAxis.y:rectAxis.y +
                                          rectAxis.h, rectAxis.x:rectAxis.x+rectAxis.w]

                    # change r to filteredRGBWithLABRules
                    isFire = 0
                    LL, LH, HL, HH = getWaveletFrequency(
                        r, wavelet_family)
                    showWaveletFrequency(
                        [LL, LH, HL, HH], isShow=False)
                    e = getEblock(LH, HL, HH)
                    # get from predict
                    isFire = predictFireOrNonFire(
                        classifier, e)

                    if isFire:
                        isFireInFrame = 1
                        drawRectangle(frame, rectAxis, text="Fire Detected")
                    isFireToString = '' if isFire else ' NF'
                    EBlock_list_pred_by_ROI.append(e)
                    isFire_list_pred_by_ROI.append(isFire)
                    ROI_fileName_list.append(
                        roi_filename+isFireToString)
                    df_test_roi.loc[df_test_roi['Video'].isin(
                        [isFireDirString+" - "+filename_dir]) & df_test_roi["ROI"].isin([roi_filename]), "label_pred"] = isFire
                    df_test_roi.loc[df_test_roi['Video'].isin(
                        [isFireDirString+" - "+filename_dir]) & df_test_roi["ROI"].isin([roi_filename]), "energy_block"] = e
                    saveOneFrame(output_loc, r_rgb, roi_filename+isFireToString,
                                 isSave=True, isRGB2BGR=True)
                    saveOneFrame(output_loc, r, 'APPLIED RULES - '+roi_filename+isFireToString,
                                 isSave=True, isRGB2BGR=True)
                    saveOneFrame(output_loc, r_colorObj, 'CO - '+roi_filename+isFireToString,
                                 isSave=True, isRGB2BGR=True)

                    # cv2.imshow('ROI original',
                    #            rS)
                    # cv2.imshow('ROI with rules',
                    #            filteredRGBWithLABRulesS)
                    # cv2.imshow('ROI with fire detected',
                    #            r_with_contoursS)

                # Save the frame in list
                n_frame = str(count+1)
                n_frame_list.append(n_frame)
                video_list.append(filename_dir)
                saveOneFrame(output_loc, frame, "Original - "+str(count+1)+" - BB ",
                             isSave=True, isRGB2BGR=True)
                isFire_list_pred_by_frame.append(isFireInFrame)
                # update label_pred

                df_test.loc[df_test['Video'].isin(
                    [isFireDirString+" - "+filename_dir]) & df_test["N-Frame"].isin([n_frame]), "label_pred"] = isFireInFrame

                saveOneFrame(output_loc, colorObj, "Color Object - "+str(count+1),
                             isSave=True, isRGB2BGR=True)
                saveOneFrame(output_loc, colorObj_WithLABRules, "Color Object with rules - "+str(count+1),
                             isSave=True, isRGB2BGR=True)

            # else:
            #     fgbgGMM = cv2.createBackgroundSubtractorMOG2(
            #         history=historyGMM, varThreshold=varTGMM, detectShadows=False)

                cv2.imshow('Original Frame', cv2.cvtColor(
                    frame_original, cv2.COLOR_RGB2BGR))
                cv2.imshow('Background Subtraction', cv2.cvtColor(
                    mask, cv2.COLOR_RGB2BGR))

                cv2.imshow('Background Subtraction in RGB', cv2.cvtColor(
                    colorObj, cv2.COLOR_RGB2BGR))
                cv2.imshow('Color Obj with Lab Rules', cv2.cvtColor(
                    colorObj_WithLABRules, cv2.COLOR_RGB2BGR))

                # cv2.imshow('Contours on Moving Object',
                #            colorObj_with_contours)
                cv2.imshow('Frame with Fire Detected', cv2.cvtColor(
                    frame, cv2.COLOR_RGB2BGR))
            # ? uncomment in sample
            if count == 0:
                time.sleep(5)
            else:
                time.sleep(0.2)
            if cv2.waitKey(1) & 0xFF == ord('q'):
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

    # ROI_fileName_list_np = np.array(ROI_fileName_list)
    # df_energy = pd.DataFrame(ROI_fileName_list_np, columns=["ROI"])
    # df_energy["energy_block"] = EBlock_list_pred_by_ROI
    # df_energy["label"] = isFire_list_pred_by_ROI
    # return df_energy


def run_video_by_dir(directory, isFire, trial, dev="Training"):

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
        main(fileName, input_loc, output_loc,
             fgbgGMM, isFireToString)
        print(video+' end')


def main_experiment_testing(directories, trial, dev):

    time_start = time.time()
    for d in directories:
        print(dev)
        print(d[0])
        run_video_by_dir(d[0], d[1], trial, dev=dev)

    # saveDataframeAsFile(df_all_energy, trial+'-'+dev +
    #                     '-label_only', trial)
    time_end = time.time()
    print("It took %d seconds in total." % (time_end-time_start))
    cv2.destroyAllWindows()


dir_testing = [
    # ['../content/drive/My Drive/DATA SET/Testing 2/Fire/', True],
    # ['../content/drive/My Drive/DATA SET/Testing 2/Video_02/', True],
    ['../content/drive/My Drive/DATA SET/Testing 2/Video_01/', True],
    # ['../content/drive/My Drive/DATA SET/Testing 2/Video_06/', True],

    # ['../content/drive/My Drive/DATA SET/Testing 2/Video_05/', True],
    # ['../content/drive/My Drive/DATA SET/Testing 2/Video_08/', True],
]


main_experiment_testing(
    dir_testing, trial+'-'+wavelet_family + '-' + svm_kernel, dev)
df_test_roi = df_test_roi.dropna(subset=['label_pred'])
df_test_roi['label_pred'] = df_test_roi['label_pred'].astype('int')
saveDataframeAsFile(df_test_roi, trial+'-ROI with pred -'+dev+'-' +
                    wavelet_family+'-'+svm_kernel, trial)
df_test = df_test.dropna(subset=['label_pred'])
df_test['label_pred'] = df_test['label_pred'].astype('int')
saveDataframeAsFile(df_test, trial+'-'+dev+'-' +
                    wavelet_family+'-'+svm_kernel, trial)

evaluationForFrame(df_test, wavelet_family + '-' + svm_kernel, trial)
evaluationForROI(df_test_roi, 'ROI - '+wavelet_family +
                 '-' + svm_kernel, trial)
