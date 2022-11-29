from utils_func import *
from classifier_func import *
from env_main import *
trial = 'Final'
dir_generated_df = '../content/drive/My Drive/A TA/generated_df/'
file_input_generated_df = dir_generated_df + trial + '/'+trial

df_training = pd.read_csv(file_input_generated_df +
                          '-'+clf_filename+'.csv')
classifier = trainData(df_training)
joblib.dump(classifier, file_input_generated_df +
            '-'+clf_filename+'-' + svm_kernel+'.pkl')

# df_test = pd.read_csv(file_input_generated_df+'-test.csv')
# svc = joblib.load(file_input_generated_df+'-'+clf_filename+'.pkl')
