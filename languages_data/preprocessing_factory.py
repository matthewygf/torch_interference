import languages_data.chinese_english_wmt_preprocessing as cwmt_preprocess

key_to_preprocess = {
  'nc-v12': cwmt_preprocess.ZhEnTranslateDatasetPreprocessor
}

def get_preprocessor(dataset_name):
  return key_to_preprocess.get(dataset_name)