from utils_func import *
from env_main import *
"""# Classifier
### SVM
"""


def trainData(df):
    # coba knn/k-means
    df_copy = copy.deepcopy(df)

    Y = df_copy['label']
    X = df_copy['energy_block']
    Y = Y.astype('int')
    X_Train = X.values.reshape(-1, 1)
    Y_Train = Y
    # Fitting the classifier into the Training set
    C = 1.0
    if svm_kernel == 'poly':
        classifier_poly = SVC(kernel=svm_kernel, degree=3, verbose=False, C=C)

        classifier_poly.fit(X_Train, Y_Train.values.ravel())

        return classifier_poly
    elif svm_kernel == 'rbf':
        classifier_poly = SVC(kernel=svm_kernel, gamma=0.7, verbose=False, C=C)

        classifier_poly.fit(X_Train, Y_Train.values.ravel())

        return classifier_poly
    else:
        classifier_linear = SVC(kernel=svm_kernel, verbose=False, C=C)

        classifier_linear.fit(X_Train, Y_Train.values.ravel())

        return classifier_linear


def testData(classifier, df):
    df_copy = copy.deepcopy(df)

    Y = df_copy['label']
    X = df_copy['energy_block']
    Y = Y.astype('int')
    # Splitting the dataset into the Training set and Test set
    X_Test = X.values.reshape(-1, 1)
    Y_Test = Y
    # Predicting the test set results
    Y_Pred = classifier.predict(X_Test)

    print(classification_report(Y_Test, Y_Pred))
    # Making the Confusion Matrix
    cm = confusion_matrix(Y_Test, Y_Pred)
    sns.heatmap(cm, annot=True)

    same_pred = np.sum(Y_Pred == Y_Test.squeeze())
    acc_train = same_pred / len(Y_Pred)
    print('Correct: ' + str(same_pred) +
          '; Incorrect: ' + str(len(Y_Pred) - same_pred))
    print('Accuracy: ' + str(acc_train * 100) + '%')
    df_copy['label_pred'] = Y_Pred

    return df_copy


def predictFireOrNonFire(clf, eblock):
    x_i = pd.Series([eblock])
    y_pred_i = clf.predict(x_i.values.reshape(-1, 1))
    return y_pred_i[0]


def perf_measure(y_act, y_p):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    y_actual = y_act.values.ravel()
    y_pred = y_p.values.ravel()
    for i in range(len(y_pred)):
        if y_actual[i] == y_pred[i] == 1:
            TP += 1
        if y_pred[i] == 1 and y_actual[i] != y_pred[i]:
            FP += 1
        if y_actual[i] == y_pred[i] == 0:
            TN += 1
        if y_pred[i] == 0 and y_actual[i] != y_pred[i]:
            FN += 1
    return TP, FP, TN, FN


def printCM(Y_Test, Y_Pred, name):
    print(name)
    print(classification_report(Y_Test, Y_Pred))
    # Making the Confusion Matrix
    cm = confusion_matrix(Y_Test, Y_Pred)
    # hmp = sns.heatmap(cm, annot=True)
    # figure = hmp.get_figure()
    # figure.savefig(out_df +
    #                'main testing sns heatmap'+name+'.png', dpi=400)
    same_pred = np.sum(Y_Pred == Y_Test.squeeze())
    acc_train = same_pred / len(Y_Pred)
    correct = same_pred
    incorrect = len(Y_Pred) - same_pred
    acc_percentage = acc_train * 100
    print('Correct: ' + str(correct) +
          '; Incorrect: ' + str(incorrect))
    print('Accuracy: ' + str(acc_percentage) + '%')
    TP, FP, TN, FN = perf_measure(Y_Test, Y_Pred)
    print('TP:', TP)
    print('FP:', FP)
    print('TN:', TN)
    print('FN:', FN)
    return TP, FP, TN, FN, acc_percentage, correct, incorrect


def evaluationForFrame(df, trial, mode):
    video_name_list = []
    TP_list = []
    FP_list = []
    TN_list = []
    FN_list = []
    acc_percentage_list = []
    correct_list = []
    incorrect_list = []

    df_copy = copy.deepcopy(df)

    # X = df_copy['Video']
    # Predicting the test set results
    # Y_Pred = classifier.predict(X_Test)
    df_copy = df_copy.dropna(subset=['label_pred'])
    Y_Pred = df_copy['label_pred']
    Y_Test = df_copy['label']
    Y_Pred = Y_Pred.astype('int')
    Y_Test = Y_Test.astype('int')
    TP_main, FP_main, TN_main, FN_main, acc_percentage_main, correct_main, incorrect_main = printCM(
        Y_Test, Y_Pred, 'main - '+trial)
    video_name_list.append('MAIN - '+trial)
    TP_list.append(TP_main)
    FP_list.append(FP_main)
    TN_list.append(TN_main)
    FN_list.append(FN_main)
    acc_percentage_list.append(acc_percentage_main)
    correct_list.append(correct_main)
    incorrect_list.append(incorrect_main)

    # # fire
    # df_fire = df_copy[~df['Video'].str.contains("Non Fire")]
    # Y_Pred_fire = df_fire['label_pred']
    # Y_Test_fire = df_fire['label']
    # Y_Pred_fire = Y_Pred_fire.astype('int')
    # Y_Test_fire = Y_Test_fire.astype('int')
    # printCM(Y_Test_fire, Y_Pred_fire, 'fire'+'-'+trial)

    # non fire
    # df_non_fire = df_copy[df['Video'].str.contains("Non Fire")]
    # Y_Pred_non_fire = df_non_fire['label_pred']
    # Y_Test_non_fire = df_non_fire['label']
    # Y_Pred_non_fire = Y_Pred_non_fire.astype('int')
    # Y_Test_non_fire = Y_Test_non_fire.astype('int')
    # printCM(Y_Test_non_fire, Y_Pred_non_fire, 'non fire')

    video_obj = df['Video'].unique()
    video_list = list(video_obj)
    for filename in video_list:
        df_by_filename = df.loc[df['Video'].isin([filename])]
        yp = df_by_filename['label_pred']
        yt = df_by_filename['label']
        yp = yp.astype('int')
        yt = yt.astype('int')
        TP, FP, TN, FN, acc_percentage, correct, incorrect = printCM(
            yt, yp, filename+'-'+trial)
        video_name_list.append(filename)
        TP_list.append(TP)
        FP_list.append(FP)
        TN_list.append(TN)
        FN_list.append(FN)
        acc_percentage_list.append(acc_percentage)
        correct_list.append(correct)
        incorrect_list.append(incorrect)
    video_name_list_np = np.array(video_name_list)
    df_evaluation = pd.DataFrame(video_name_list_np, columns=["Video name"])
    df_evaluation["TP"] = TP_list
    df_evaluation["FP"] = FP_list
    df_evaluation["TN"] = TN_list
    df_evaluation["FN"] = FN_list
    df_evaluation["Accuracy"] = acc_percentage_list
    df_evaluation["Correct"] = correct_list
    df_evaluation["Incorrect"] = incorrect_list

    saveDataframeAsFile(df_evaluation, 'evaluation-'+trial, mode)

    return df_copy


def evaluationForROI(df, trial, mode):
    video_name_list = []
    TP_list = []
    FP_list = []
    TN_list = []
    FN_list = []
    acc_percentage_list = []
    correct_list = []
    incorrect_list = []

    df_copy = copy.deepcopy(df)

    # X = df_copy['Video']
    # Predicting the test set results
    # Y_Pred = classifier.predict(X_Test)
    df_copy = df_copy.dropna(subset=['label_pred'])
    Y_Pred = df_copy['label_pred']
    Y_Test = df_copy['label']
    Y_Pred = Y_Pred.astype('int')
    Y_Test = Y_Test.astype('int')
    TP_main, FP_main, TN_main, FN_main, acc_percentage_main, correct_main, incorrect_main = printCM(
        Y_Test, Y_Pred, 'main - '+trial)
    video_name_list.append('MAIN - '+trial)
    TP_list.append(TP_main)
    FP_list.append(FP_main)
    TN_list.append(TN_main)
    FN_list.append(FN_main)
    acc_percentage_list.append(acc_percentage_main)
    correct_list.append(correct_main)
    incorrect_list.append(incorrect_main)

    # # fire
    # df_fire = df_copy[~df['Video'].str.contains("Non Fire")]
    # Y_Pred_fire = df_fire['label_pred']
    # Y_Test_fire = df_fire['label']
    # Y_Pred_fire = Y_Pred_fire.astype('int')
    # Y_Test_fire = Y_Test_fire.astype('int')
    # printCM(Y_Test_fire, Y_Pred_fire, 'fire'+'-'+trial)

    # non fire
    # df_non_fire = df_copy[df['Video'].str.contains("Non Fire")]
    # Y_Pred_non_fire = df_non_fire['label_pred']
    # Y_Test_non_fire = df_non_fire['label']
    # Y_Pred_non_fire = Y_Pred_non_fire.astype('int')
    # Y_Test_non_fire = Y_Test_non_fire.astype('int')
    # printCM(Y_Test_non_fire, Y_Pred_non_fire, 'non fire')

    video_obj = df['Video'].unique()
    video_list = list(video_obj)
    for filename in video_list:
        df_by_filename = df.loc[df['Video'].isin([filename])]
        yp = df_by_filename['label_pred']
        yt = df_by_filename['label']
        yp = yp.astype('int')
        yt = yt.astype('int')
        TP, FP, TN, FN, acc_percentage, correct, incorrect = printCM(
            yt, yp, filename+'-'+trial)
        video_name_list.append(filename)
        TP_list.append(TP)
        FP_list.append(FP)
        TN_list.append(TN)
        FN_list.append(FN)
        acc_percentage_list.append(acc_percentage)
        correct_list.append(correct)
        incorrect_list.append(incorrect)
    video_name_list_np = np.array(video_name_list)
    df_evaluation = pd.DataFrame(video_name_list_np, columns=["Video name"])
    df_evaluation["TP"] = TP_list
    df_evaluation["FP"] = FP_list
    df_evaluation["TN"] = TN_list
    df_evaluation["FN"] = FN_list
    df_evaluation["Accuracy"] = acc_percentage_list
    df_evaluation["Correct"] = correct_list
    df_evaluation["Incorrect"] = incorrect_list

    saveDataframeAsFile(df_evaluation, 'evaluation-'+trial, mode)

    return df_copy


def plotOneDimension(df_ori, dir_filename, dev="training"):
    df = copy.deepcopy(df_ori)

    target_names = ['-1', '+1']
    print(df.keys())
    X1 = df['energy_block']
    X2 = np.ones((len(df), 1), int)
    X_training = np.array(list(zip(X1, X2)))
    # X_training=np.array(np.transpose([X1])) # alternative way, but you cannot plot
    ylabel = 'label' if dev == "training" else "label_pred"
    y_training = df[ylabel]

    idxPlus = y_training[y_training < 0].index
    idxMin = y_training[y_training > 0].index
    plt.scatter(X_training[idxPlus, 0], X_training[idxPlus, 1], c='b', s=50)
    plt.scatter(X_training[idxMin, 0], X_training[idxMin, 1], c='r', s=50)
    plt.legend(target_names, loc=3)
    plt.xlabel('Energy Block')
    plt.ylabel('X2'+"-"+dev)
    plt.savefig(dir_filename+'-'+dev+'.png')


def plotOneDimensionNF(df_ori, dir_filename, dev="training"):
    df = copy.deepcopy(df_ori)

    target_names = ['-1', '+1']
    print(df.keys())
    X1 = df['energy_block']
    X2 = np.ones((len(df), 1), int)
    X_training = np.array(list(zip(X1, X2)))
    # X_training=np.array(np.transpose([X1])) # alternative way, but you cannot plot
    ylabel = 'label' if dev == "training" else "label_pred"
    y_training = df[ylabel]

    idxPlus = y_training[y_training < 0].index
    idxMin = y_training[y_training > 0].index
    plt.scatter(X_training[idxPlus, 0], X_training[idxPlus, 1], c='b', s=50)
    # plt.scatter(X_training[idxMin, 0], X_training[idxMin, 1], c='r', s=50)
    plt.legend(target_names, loc=3)
    plt.xlabel('Energy Block')
    plt.ylabel('X2'+"-"+dev)
    plt.savefig(dir_filename+'-'+dev+' NF.png')
