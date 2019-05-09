import os
import codecs
import jieba
import csv
jieba.initialize()
from allennlp.common.file_utils import cached_path

def count_lines(path):
  count = 0
  with codecs.open(path, "r", "utf-8") as f:
    for line in f:
      count += 1
  return count 

def create_files(origin_path, cached_path, train_count, total_count):
  file_name = os.path.basename(origin_path)
  splits = os.path.splitext(file_name)
  train_out = splits[0]+".train"+splits[1]
  val_out = splits[0]+".val"+splits[1]
  count = 0
  with open(cached_path, 'r', encoding='UTF-8') as c:
    with open(train_out, 'a+', encoding='UTF-8') as t:
      with open(val_out, 'a+', encoding='UTF-8') as v:
          for line in c:
            if len(line.strip()) == 0: continue
            if count < train_count:
              t.write(line.replace('）','').replace('   ','').replace('；','').replace('\xe2','').replace('   ','').replace('（','').replace(' ）','').replace('  （', '').replace('%','').replace('% ','').replace('�','').replace('，','').replace('– ','').replace('’','').replace('"','').replace('\'', '').replace('—', '').replace(')','').replace('？', '').replace('?','').replace('、','').replace('$','').replace(' [','').replace('[','').replace(']','').replace('; ','').replace('.','').replace(' ‘','').replace('！','').replace('(','').replace('“','').replace('”', '').replace(',','').replace('：','').replace('。', '').replace('   ',''))
            else:
              v.write(line.replace('）','').replace('   ','').replace('；','').replace('\xe2','').replace('   ','').replace('（','').replace(' ）','').replace('  （', '').replace('%','').replace('% ','').replace('�','').replace('，','').replace('– ','').replace('’','').replace('"','').replace('\'', '').replace('—', '').replace(')','').replace('？', '').replace('?','').replace('、','').replace('$','').replace(' [','').replace('[','').replace(']','').replace('; ','').replace('.','').replace(' ‘','').replace('！','').replace('(','').replace('“','').replace('”', '').replace(',','').replace('：','').replace(' 。','').replace('   ',''))
            count += 1

  print(train_count)

def main():
  dataset = 'nc_zhen17'
  dataset_dir = 'data/'
  filepaths = [
    'news-commentary-v14.en-zh.val.tsv',
  ]
  train_val_split = 0.1
  cached_dataset_dir = os.path.join(dataset_dir, dataset)
  # cached_paths = []
  # for fp in filepaths:
  #   cached_paths.append((fp, cached_path(fp, cached_dataset_dir)))
  problem_line = []
  # for fp, cp in cached_paths:
    # total_count = count_lines(cp)
    # train_count = total_count * (1-train_val_split)
    # create_files(fp, cp, train_count, total_count)
  with open(filepaths[0], "r", encoding='UTF-8') as data_file:
    for line_num, row in enumerate(csv.reader(data_file, delimiter="\t")):
        if len(row) != 2:
          problem_line.append(line_num + 1)
            # raise ConfigurationError("Invalid line format: %s (line number %d)" % (row, line_num + 1))
  print(len(problem_line))

if __name__ == "__main__":
  main()