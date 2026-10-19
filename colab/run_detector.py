# Fabric defect DETECTOR (YOLO11-OBB): hole / stain / broken_thread / irregular_yarn
# Run inside Colab: the bootstrap notebook exec()s this file. Needs a T4 GPU. ~60-90 min.
#  * oriented boxes keep thin diagonal thread defects tight (TILDA's native format); own boxes become rotated boxes with angle 0
#  * trained on grayscale images: convert live frames to gray before inference
#  * no 'normal' class (as in the team plan): clean fabric is simply background
import os, sys, subprocess, zipfile, shutil, glob

subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'ultralytics'], check=True)
import torch
assert torch.cuda.is_available(), 'No GPU! Runtime > Change runtime type > T4 GPU, then run again.'
print('GPU:', torch.cuda.get_device_name(0))

D = '/content/drive/MyDrive/fabric_models'
TAG = 'det'

# 1) download the two public sources straight into Colab (~900 MB)
SRC = {'bd': ('h2ytgy2pmb', 699967798, 'Bangladesh fabric set (Hole / Broken End / Oil)'),
       'tilda': ('8y46pm2m6y', 193352808, 'TILDA-OBB (oriented boxes)')}
os.makedirs('/content/src', exist_ok=True)
for key, (did, size, label) in SRC.items():
    z = f'/content/src/{key}.zip'
    if not (os.path.exists(z) and os.path.getsize(z) == size):
        print('downloading', label)
        subprocess.run(['curl', '-L', '-s', '--retry', '6', '--retry-delay', '4', '-C', '-', '-o', z,
                        f'https://data.mendeley.com/public-api/zip/{did}/download/1'], check=True)
    assert os.path.getsize(z) == size, f'{key}: incomplete download ({os.path.getsize(z)} of {size}) - run again'
    out = f'/content/src/{key}'
    if not os.path.isdir(out):
        with zipfile.ZipFile(z) as zf:
            bad = [n for n in zf.namelist() if n.startswith('/') or '..' in n.split('/')]
            assert not bad, f'unsafe paths in zip: {bad[:3]}'
            zf.extractall(out)
    print('ready:', label)
TILDA_ROOT = next(d for d, _, _ in os.walk('/content/src/tilda') if os.path.basename(d) == 'TILDA-OBB-ALL')

# 2) build the datasets (grayscale, leak-free splits) with my verified labels
sys.path.insert(0, D)
import importlib, build_dataset
importlib.reload(build_dataset)
build_dataset.build('/content/src/bd', TILDA_ROOT, f'{D}/label_pack.json.gz', '/content/data', gray=True)

# 3) train
from ultralytics import YOLO
model = YOLO('yolo11s-obb.pt')
model.train(data='/content/data/det/data.yaml', epochs=80, imgsz=640, batch=16, patience=20, cos_lr=True,
            degrees=0, fliplr=0.5, flipud=0.5, scale=0.4, translate=0.1, mosaic=1.0, close_mosaic=10,
            hsv_h=0, hsv_s=0, hsv_v=0.35,
            project='/content/runs', name=TAG, exist_ok=True, plots=True, seed=0)

# 4) overfitting check
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv(f'/content/runs/{TAG}/results.csv'); df.columns = [c.strip() for c in df.columns]
fig, ax = plt.subplots(1, 3, figsize=(16, 4))
for a, (t, v) in zip(ax, [('train/box_loss', 'val/box_loss'), ('train/cls_loss', 'val/cls_loss')]):
    a.plot(df[t], label='train'); a.plot(df[v], label='val'); a.legend(); a.set_title(t.split('/')[1])
ax[2].plot(df['metrics/mAP50(B)'], label='mAP50'); ax[2].plot(df['metrics/mAP50-95(B)'], label='mAP50-95'); ax[2].legend(); ax[2].set_title('val mAP'); plt.show()

# 5) held-out TEST: all images, then own-camera images only, then TILDA only
best = YOLO(f'/content/runs/{TAG}/weights/best.pt')
names = ['hole', 'stain', 'broken_thread', 'irregular_yarn']
for tag, yaml in [('ALL test images', 'data.yaml'), ('OWN-camera test images only', 'data_own.yaml'),
                  ('TILDA test images only', 'data_tilda.yaml')]:
    m = best.val(data=f'/content/data/det/{yaml}', split='test', plots=(tag == 'ALL test images'), verbose=False,
                 project='/content/runs', name='test_' + yaml.split('.')[0], exist_ok=True)
    print(f'\n== {tag}:  mAP50={m.box.map50:.3f}  mAP50-95={m.box.map:.3f}  P={m.box.mp:.3f}  R={m.box.mr:.3f}')
    for i, c in enumerate(m.box.ap_class_index):
        print(f'   {names[int(c)]:15s} P={m.box.p[i]:.3f} R={m.box.r[i]:.3f} AP50={m.box.ap50[i]:.3f}')

# 6) look at predictions on test images
import random
from PIL import Image
from IPython.display import display
random.seed(1)
fs = random.sample(glob.glob('/content/data/det/images/test/*.jpg'), 12)
res = best.predict(fs, conf=0.25, verbose=False)
sheet = Image.new('RGB', (420 * 4, 300 * 3))
for i, r in enumerate(res):
    sheet.paste(Image.fromarray(r.plot()[..., ::-1]).resize((420, 300)), ((i % 4) * 420, (i // 4) * 300))
display(sheet)

# 7) save to Drive
dst = f'{D}/{TAG}'
os.makedirs(dst, exist_ok=True)
for f in ['weights/best.pt', 'weights/last.pt', 'results.csv', 'results.png', 'confusion_matrix.png', 'confusion_matrix_normalized.png', 'args.yaml']:
    p = f'/content/runs/{TAG}/{f}'
    if os.path.exists(p):
        shutil.copy(p, dst)
print('SAVED to Drive:', dst, '->', sorted(os.listdir(dst)))