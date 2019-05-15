import multiprocessing
import subprocess
import os
import time




def clean_timeline(args):
  path, filename = args
  sets = [
            ['googlenet_cmd'],
            ['mobilenetv2_cmd'],
            ['vgg19_cmd'],
            ['pos_cmd'],
            ['mt1_cmd'],
            ['mt2_cmd'],
            ['googlenet_cmd','googlenet_cmd'],
            ['mobilenetv2_cmd','mobilenetv2_cmd'],
            ['pos_cmd', 'pos_cmd'],
            ['googlenet_cmd', 'mobilenetv2_cmd'],
            ['googlenet_cmd', 'vgg19_cmd'],
            ['googlenet_cmd', 'pos_cmd'],
            ['googlenet_cmd', 'mt1_cmd'],
            ['mobilenetv2_cmd', 'vgg19_cmd'],
            ['mobilenetv2_cmd', 'pos_cmd'],
            ['mobilenetv2_cmd', 'mt1_cmd'],
	          ['mobilenetv2_cmd', 'mt2_cmd'] 
           ]
  try:
    # clean "==="
    path_dir = int(os.path.basename(os.path.dirname(path)))
    new_name = "_".join(sets[path_dir])
    nvprof_exp_dir = os.path.dirname(path)
    clean_timeline_path = os.path.join(nvprof_exp_dir, filename+"_"+new_name+"_new.csv")
    with open(path, 'r') as org_newtimeline:
      with open(clean_timeline_path, 'a+') as clean_newtimeline:
        for line in org_newtimeline:
          if "===" in line: continue
          clean_newtimeline.write(line)
    os.remove(path)
  except:
    print ("path %s error" % path)
  

def main():
  # clean "==="
  timeline_to_clean = []
  for root_dir, dirs, files in os.walk(os.getcwd()):
    for filename in files:
      if "__timeline" in filename and "experiment" in root_dir:
        timeline_path = os.path.join(root_dir, filename)
        nvprof_dirs = os.path.join(os.getcwd(), 'nvprof_conv')
        out_log = os.path.join(nvprof_dirs, 'convs.log')
        path_dir = int(os.path.basename(os.path.dirname(filename)))
        nvprof_exp_dir = os.path.join(nvprof_dirs, path_dir)
        if not os.path.exists(nvprof_exp_dir):
          os.makedirs(nvprof_exp_dir)
        new_timeline_path = os.path.join(nvprof_exp_dir, filename+"_conv.csv")
        with open(out_log, 'a+') as outlogs_handle:
          # only one nvprof each time.
          conv_p = subprocess.Popen(['nvprof', '--print-gpu-trace', '--csv', '-i', timeline_path, '--log-file', new_timeline_path], stderr=outlogs_handle, stdout=outlogs_handle)
          while conv_p.poll() is None:
            time.sleep(10)
          print("done converting %s" % timeline_path)
          timeline_to_clean.append((new_timeline_path, filename))
  # multiprocess clean
  with multiprocessing.Pool(processes=4) as pools:
    pools.map(clean_timeline, timeline_to_clean)

if __name__ == "__main__":
  main()