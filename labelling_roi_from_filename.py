from utils_func import *
from env_main import *


def run_roi_by_dir(directory, isFire, trial, dev="Training"):
    df_energy_in_main = pd.DataFrame(
        columns=["Video", "ROI", "label"])
    # listDirectory = list(os.listdir(directory))
    # name_list = os.listdir(directory)
    # full_list = [os.path.join(directory, i) for i in name_list]
    # # time sorted
    # listDirectory = sorted(full_list, key=os.path.getmtime)

    name_list = os.listdir(directory)
    listDirectory = [os.path.basename(i) for i in name_list]

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
        # try:
        #     os.makedirs(output_loc)
        # except OSError as e:
        #     if e.errno != errno.EEXIST:
        #         raise
        #         # time.sleep might help here
        #     pass

        listROI = list(os.listdir(input_loc))
        N_ROI = len(listROI)

        video_list = []
        isFire_list_pred_by_frame = []
        n_roi_list = []

        for j in range(N_ROI):
            frame_fileName = listROI[j]
            fileName, ext = os.path.splitext(frame_fileName)
            isFireInFrame = 1
            if "NF" in frame_fileName:
                isFireInFrame = 0
            print(frame_fileName, isFireInFrame)
            n_roi = fileName.replace(" NF", "")
            n_roi_list.append(n_roi)
            video_list.append(isFireToString+" - "+video)
            isFire_list_pred_by_frame.append(isFireInFrame)

        video_list_np = np.array(video_list)
        df_energy = pd.DataFrame(video_list_np, columns=["Video"])
        df_energy["ROI"] = n_roi_list
        df_energy["label"] = isFire_list_pred_by_frame
        df_energy_in_main = df_energy_in_main.append(df_energy)
        print(video+' end')
    return df_energy_in_main


def main_labelling(directories, trial, dev):
    time_start = time.time()
    df_all_energy = pd.DataFrame(
        columns=["Video", "ROI", "label"])
    for d in directories:
        print(dev)
        print(d[0])
        df_energy_in_main = run_roi_by_dir(d[0], d[1], trial, dev=dev)
        df_all_energy = df_all_energy.append(df_energy_in_main)
    saveDataframeAsFile(df_all_energy, trial+'-'+dev, trial)
    joblib.dump(df_all_energy, out_df+trial+'-'+dev+'.pkl')
    time_end = time.time()
    print("It took %d seconds in total." % (time_end-time_start))


dir_roi_with_label = [
    ['../content/drive/My Drive/A TA/evaluation/Final-actual-label/Fire/', True],
    # ['../content/drive/My Drive/A TA/evaluation/Final-actual-label-2/Fire/', True],
    # ['../content/drive/My Drive/A TA/frame_testing_with_label/Non Fire/', False],
]

main_labelling(dir_roi_with_label, trial,
               roi_testing_with_label)
