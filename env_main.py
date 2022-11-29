trial = 'Final'
wavelet_training_name = 'Wavelet Energy Training'
wavelet_testing_name = 'Wavelet Energy Training'
# db5, sym4, bior3.5
wavelet_family = 'sym4'
#linear, rbf, poly
svm_kernel = 'poly'
dir_generated_df = '../content/drive/My Drive/A TA/generated_df/'
file_input_generated_df = dir_generated_df + trial + '/'+trial

clf_filename = wavelet_training_name + '-' + wavelet_family
dev = "main testing"
out_df = '../content/drive/My Drive/A TA/generated_df/'+trial+'/'
dir_testing_with_label = "Frame Testing -- with label"
roi_testing_with_label = "ROI Testing -- with label"
