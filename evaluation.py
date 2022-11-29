from env_main import *
from classifier_func import *
# df_test = pd.read_csv(file_input_generated_df+'-' + dev+'-' +
#   wavelet_family+'-'+svm_kernel+'.csv')

# evaluationForFrame(df_test, wavelet_family + '-' + svm_kernel, trial)


df_test_roi = pd.read_csv(file_input_generated_df+'-' + 'ROI with pred -'+dev+'-' +
                          wavelet_family+'-'+svm_kernel+'.csv')
df_test_roi = df_test_roi.dropna(subset=['label_pred'])
df_test_roi['label_pred'] = df_test_roi['label_pred'].astype('int')
evaluationForROI(df_test_roi, 'ROI - '+wavelet_family +
                 '-' + svm_kernel, trial)

# df_training = pd.read_csv(file_input_generated_df +
#                           '-'+wavelet_training_name+'-'+wavelet_family+'.csv')
# plotOneDimension(df_training, out_df+'1D-plot', dev="training")

# plotOneDimensionNF(df_training, out_df+'1D-plot', dev="training")
