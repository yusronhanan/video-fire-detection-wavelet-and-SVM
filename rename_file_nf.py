from utils_func import *
from env_main import *


def run_frame_by_dir(directory, isFire, trial, dev="Training"):
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

        for j in range(N_ROI):
            # if j != 0:
            frame_fileName = listROI[j]
            fileName, ext = os.path.splitext(frame_fileName)
            # if 'NF' not in fileName:
            os.rename(input_loc+'/'+frame_fileName,
                        input_loc+'/'+fileName+' NF'+ext)
            # isFireInFrame = 1
            # if "NF" in frame_fileName:
            #     isFireInFrame = 0
            # print(frame_fileName, isFireInFrame)
            # n_frame = fileName.replace(
            #     "Original - ", "").replace(" NF", "")
            # n_frame_list.append(n_frame)
            # video_list.append(video)
            # isFire_list_pred_by_frame.append(isFireInFrame)

        print(video+' end')


def main_rename(directories, trial, dev):
    time_start = time.time()
    for d in directories:
        print(dev)
        print(d[0])
        run_frame_by_dir(d[0], d[1], trial, dev=dev)

    time_end = time.time()
    print("It took %d seconds in total." % (time_end-time_start))


dir_rename = [
    ['../content/drive/My Drive/A TA/rename_to_NF/', True],
]

main_rename(dir_rename, trial,
            dir_testing_with_label)
