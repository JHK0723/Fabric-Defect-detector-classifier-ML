# Fabric defect CLASSIFIER (YOLO11-cls): hole / stain / broken_thread / irregular_yarn / non_defective
# Run inside Colab: the bootstrap notebook exec()s this file. Needs a T4 GPU. ~15-25 min.
#  * data: 256x256 patches cut from labelled images (own Bangladesh set + TILDA-OBB); non_defective patches come from
#    clean regions of the SAME images, so the model cannot cheat by recognising the camera or fabric
#  * everything is grayscale (colour would reveal the source dataset); convert frames to gray before inference
#  * splits keep neighbouring photos together; results are reported per source so domain shift is visible
import os, sys, subprocess, zipfile, shutil, glob

subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'ultralytics', 'scikit-learn'], check=True)
import torch
assert torch.cuda.is_available(), 'No GPU! Runtime > Change runtime type > T4 GPU, then run again.'
print('GPU:', torch.cuda.get_device_name(0))

D = '/content/drive/MyDrive/fabric_models'
TAG = 'cls'

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
model = YOLO('yolo11s-cls.pt')
model.train(data='/content/data/cls', epochs=45, imgsz=224, batch=64, patience=10, optimizer='AdamW', lr0=0.0006,
            weight_decay=0.01, label_smoothing=0.1, dropout=0.3,
            degrees=0, fliplr=0.5, flipud=0.5, scale=0.3, translate=0.1, hsv_h=0, hsv_s=0, hsv_v=0.35,
            erasing=0.2, auto_augment=None,        # no rotation: a vertical line (broken_thread) must stay vertical
            project='/content/runs', name=TAG, exist_ok=True, plots=True, seed=0, verbose=False)

# 4) overfitting check
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv(f'/content/runs/{TAG}/results.csv'); df.columns = [c.strip() for c in df.columns]
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(df['train/loss'], label='train'); ax[0].plot(df['val/loss'], label='val'); ax[0].legend(); ax[0].set_title('loss')
ax[1].plot(df['metrics/accuracy_top1']); ax[1].set_title('val top-1'); plt.show()

# 5) held-out TEST: overall, per class, per source
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
best = YOLO(f'/content/runs/{TAG}/weights/best.pt')
rows = []
for f in sorted(glob.glob('/content/data/cls/test/*/*.jpg')):
    r = best(f, verbose=False)[0]
    rows.append((os.path.basename(os.path.dirname(f)), r.names[r.probs.top1],
                 'tilda' if os.path.basename(f).startswith('tilda_') else 'own'))
d = pd.DataFrame(rows, columns=['true', 'pred', 'source'])
print('TEST top-1 accuracy: %.3f  (n=%d)\n' % ((d.true == d.pred).mean(), len(d)))
print(classification_report(d.true, d.pred, digits=3))
print('accuracy by source:'); print(d.assign(ok=d.true == d.pred).groupby('source').ok.agg(['mean', 'count']).round(3))
print('\nrecall by class x source:')
print(d.assign(ok=d.true == d.pred).pivot_table(index='true', columns='source', values='ok', aggfunc=['mean', 'count']).round(2))
labels = sorted(d.true.unique()); cm = confusion_matrix(d.true, d.pred, labels=labels, normalize='true')
plt.figure(figsize=(6, 5)); plt.imshow(cm, cmap='Blues'); plt.xticks(range(len(labels)), labels, rotation=45, ha='right'); plt.yticks(range(len(labels)), labels)
for i in range(len(labels)):
    for j in range(len(labels)):
        plt.text(j, i, f'{cm[i, j]:.2f}', ha='center', va='center')
plt.title('test confusion (row-normalised)'); plt.tight_layout(); plt.show()

# 6) save to Drive
dst = f'{D}/{TAG}'
os.makedirs(dst, exist_ok=True)
for f in ['weights/best.pt', 'weights/last.pt', 'results.csv', 'results.png', 'confusion_matrix.png', 'confusion_matrix_normalized.png', 'args.yaml']:
    p = f'/content/runs/{TAG}/{f}'
    if os.path.exists(p):
        shutil.copy(p, dst)
print('SAVED to Drive:', dst, '->', sorted(os.listdir(dst)))