from wavelet_func import *
from classifier_func import *
from env_main import *


def main_wavelet(input_loc, isFire):
    # Log the time
    time_start = time.time()

    count = 0
    count_Fire = 0
    count_NonFire = 0

    print("Calculate energy block by using wavelet..\n")
    EBlock_list = []
    isFire_list = []
    ROI_fileName_list = []
    listROI = list(os.listdir(input_loc))
    N_ROI = len(listROI)
    for i in range(N_ROI):
        ROI_fileName = listROI[i]
        if ROI_fileName == '.DS_Store' or "APPLIED RULES" not in ROI_fileName:
            continue
        # print(ROI_fileName+' start')
        fileName, ext = os.path.splitext(ROI_fileName)
        input_ROI = input_loc+'/'+fileName+ext
        r_bgr = cv2.imread(input_ROI)
        r = cv2.cvtColor(r_bgr, cv2.COLOR_BGR2RGB)

        # define fire / non fire
        isFire = 1
        if "NF" in ROI_fileName:
            isFire = 0
            count_NonFire = count_NonFire + 1
        else:
            count_Fire = count_Fire + 1
        LL, LH, HL, HH = getWaveletFrequency(r, wavelet_family)
        showWaveletFrequency([LL, LH, HL, HH], isShow=False)
        e = getEblock(LH, HL, HH)

        EBlock_list.append(e)
        isFire_list.append(isFire)
        ROI_fileName_list.append(input_ROI)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        count = count + 1
    # Log the time again
    time_end = time.time()
    # Print stats
    print("Done get energy block of.\n%d ROI filtered" % count)
    print("%d ROI is actual fire" % count_Fire)
    print("\n%d ROI is not actual fire" % count_NonFire)

    print("It took %d seconds forconversion." % (time_end-time_start))

    ROI_fileName_list_np = np.array(ROI_fileName_list)
    df_energy = pd.DataFrame(ROI_fileName_list_np, columns=["ROI"])
    df_energy["energy_block"] = EBlock_list
    df_energy["label"] = isFire_list
    return df_energy


def run_roi_by_dir(directory, isFire, trial, dev="Training"):
    df_energy_in_main = pd.DataFrame(columns=["ROI", "energy_block", "label"])

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
        if video == '.DS_Store' or 'NOT' in video:
            continue
        print(video+' start')
        fileName, ext = os.path.splitext(video)
        input_loc = directory+fileName+ext
        # output_loc = '../content/drive/My Drive/A TA/' + \
        #     dev+'/'+trial+'/'+isFireToString+'/'+fileName
        # try:
        #     os.makedirs(output_loc)
        # except OSError as e:
        #     if e.errno != errno.EEXIST:
        #         raise
        #         # time.sleep might help here
        #     pass

        df_energy = main_wavelet(input_loc, isFire)
        df_energy_in_main = df_energy_in_main.append(df_energy)
        print(video+' end')
    return df_energy_in_main


def main(directories, trial, dev):
    df_all_energy = pd.DataFrame(columns=[
                                 "ROI", "energy_block", "label"])
    time_start = time.time()
    for d in directories:
        print(dev)
        print(d[0])
        df_energy_in_main = run_roi_by_dir(d[0], d[1], trial, dev=dev)
        df_all_energy = df_all_energy.append(df_energy_in_main)
    saveDataframeAsFile(df_all_energy, trial+'-'+dev+'-'+wavelet_family, trial)
    time_end = time.time()
    print("It took %d seconds in total." % (time_end-time_start))
    return df_all_energy


dir_training = [
    ['../content/drive/My Drive/A TA/ROI from GMM Training with rules/'+trial+'/Fire/', True],
    ['../content/drive/My Drive/A TA/ROI from GMM Training with rules/' +
        trial+'/Non Fire/', False],
]
df_training = main(dir_training, trial, "Wavelet Energy Training")
# df_training = pd.read_csv(file_input_generated_df +
#                           '-'+wavelet_training_name+'.csv')

plotOneDimension(df_training, out_df+'1D-plot' +
                 '-'+wavelet_family, dev="training")
