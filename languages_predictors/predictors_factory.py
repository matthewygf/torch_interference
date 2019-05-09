from allennlp.predictors import SentenceTaggerPredictor, SimpleSeq2SeqPredictor

from languages_predictors import universal_eng_pos_predictor

key_to_predictors = {
  'debug': SentenceTaggerPredictor,
  'ud-eng': universal_eng_pos_predictor.UniversalEngPosPredictor,
  'nc_zhen': SimpleSeq2SeqPredictor
}

predict_logits = {
  'pos': 'tag_logits',
  'nc_zhen': 'predicted_tokens'
}

def get_predictors(dataset_name, model, data_reader):
  predictor = key_to_predictors[dataset_name](model, dataset_reader=data_reader)
  return predictor

def get_logits_key(task_name):
  return predict_logits.get(task_name)