# casia2015: 1,050,000 lines
# casict2015: 2,036,833 lines
# datum2015:  1,000,003 lines
# datum2017: 1,999,968 lines
# NEU2017:  2,000,000 lines 
#TODO:
_CWMT_TRAIN_A_DATASETS = {
  'casia2015': ['casia2015_ch.txt', 'casia2015_en.txt'],
  'casict2015': ['casict2015_ch.txt', 'casict2015_en.txt'],
  'neu2017': ['NEU_cn.txt', 'NEU_en.txt'],
  'datum2015': ['datum_ch.txt', 'datum_en.txt']
  # TODO: there is more.
}

class ZhEnTranslateDatasetPreprocessor(object):
  def __init__(self, path_to_files, ):
    return super().__init__(*args, **kwargs)

  #https://github.com/twairball/t2t_wmt_zhen/blob/master/data_generators/utils.py
  def _preprocess_sgm(line, is_sgm):
    """preprocessing to strip tags in SGM files"""
    pass