from cie_lab_func import *
from gmm_func import *
from env_main import *


def main_cie_lab(input_loc, output_loc, isFire):
    # Log the time
    time_start = time.time()

    count = 0
    count_CF = 0
    print("Applying cie lab rules..\n")
    ROI_cf_list_loc = []
    filteredRGBWithLABRulesToSave = []
    filteredRGBWithLABRules_original_ToSave = []
    listROI = list(os.listdir(input_loc))
    N_ROI = len(listROI)
    for i in range(N_ROI):
        ROI_fileName = listROI[i]
        if '.DS_Store' in ROI_fileName or 'Mask' in ROI_fileName or 'Original' in ROI_fileName or 'Color Object' in ROI_fileName:
            continue
        # print(ROI_fileName+' start')
        fileName, ext = os.path.splitext(ROI_fileName)
        input_ROI = input_loc+'/'+fileName+ext
        r_bgr = cv2.imread(input_ROI)
        r = cv2.cvtColor(r_bgr, cv2.COLOR_BGR2RGB)

        # get candidate fire
        filteredRGBWithLABRules = getCandidateFire(r)
        if not isAllPixelBlack(filteredRGBWithLABRules):
            ROI_list_in_RGB_with_lab_rules, ROI_rectangle_list_in_RGB_with_lab_rules, contours_ROI_list_in_RGB_with_lab_rules = getObjectList(
                filteredRGBWithLABRules, filteredRGBWithLABRules, MIN_SIZE_FOR_MOVEMENT=1000)

            if len(ROI_list_in_RGB_with_lab_rules) > 0:
                count_CF = count_CF+1
                ROI_cf_list_loc.append(input_ROI)
                isFireToString = '' if isFire else ' NF'
                saveOneFrame(output_loc, r, fileName+isFireToString,
                             isSave=True, isRGB2BGR=True)
                saveOneFrame(output_loc, filteredRGBWithLABRules, 'APPLIED RULES - '+fileName+isFireToString,
                             isSave=True, isRGB2BGR=True)

                filteredRGBWithLABRulesToSave.append(
                    filteredRGBWithLABRules)
                filteredRGBWithLABRules_original_ToSave.append(r)
                # drawRectangle(frame, rectAxis)
                # showFrame(filteredRGBWithLABRules,
                #           title="RGB with applied lab rules", isShow=False)
                filteredRGBWithLABRules_copy = copy.deepcopy(
                    filteredRGBWithLABRules)
                filteredRGBWithLABRulesS = cv2.resize(cv2.cvtColor(
                    filteredRGBWithLABRules_copy, cv2.COLOR_RGB2BGR), (960, 540))
                cv2.imshow('ROI with rules',
                           filteredRGBWithLABRulesS)

                r_copy = copy.deepcopy(r)
                rS = cv2.resize(cv2.cvtColor(
                    r_copy, cv2.COLOR_RGB2BGR), (960, 540))
                cv2.imshow('ROI original',
                           rS)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        count = count + 1
    # Log the time again
    time_end = time.time()
    # Print stats
    print("Done get ROI from applying rules of.\n%d ROI filtered" % count)
    print("Done get ROI from applying rules of.\n%d ROI is candidate fire" % count_CF)
    print("It took %d seconds forconversion." % (time_end-time_start))


def run_roi_by_dir(directory, isFire, trial, dev="Training"):
    listDirectory = list(os.listdir(directory))
    isFireToString = 'Fire' if isFire else 'Non Fire'
    # change limit, limit > 0, no limit otherwise
    # print(listDirectory)
    limit = -1
    N = len(listDirectory)
    if limit > 0:
        N = limit
    for i in range(N):
        # for video in listDirectory:
        video = listDirectory[i]
        if '.DS_Store' in video:
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

        main_cie_lab(input_loc, output_loc, isFire)
        print(video+' end')


def main_experiment(directories, trial, dev):
    time_start = time.time()
    for d in directories:
        print(dev)
        print(d[0])
        run_roi_by_dir(d[0], d[1], trial, dev=dev)
    time_end = time.time()
    print("It took %d seconds in total." % (time_end-time_start))


# remove later
trial = 'Final'
# dir_training = [
#     ['../content/drive/My Drive/A TA/ROI from GMM Training/'+trial+'/Fire/', True],
#     ['../content/drive/My Drive/A TA/ROI from GMM Training/' +
#         trial+'/Non Fire/', False],
# ]

# main_experiment(dir_training, trial,
#                 "ROI from GMM Training with rules")

dir_testing = [
    ['../content/drive/My Drive/A TA/ROI from GMM Testing/'+trial+'/Fire/', True],
    # ['../content/drive/My Drive/A TA/ROI from GMM Testing/Non Fire/', False],
]
main_experiment(dir_testing, trial,
                "ROI from GMM Testing with rules")
