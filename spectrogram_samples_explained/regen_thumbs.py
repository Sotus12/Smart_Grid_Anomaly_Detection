import os
import csv
import sys
import subprocess

# Ensure Pillow is available
try:
    from PIL import Image
except Exception:
    print('Pillow not found; installing...')
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'Pillow'])
    from PIL import Image

base = r"C:\SOFTWARE\DL Project\spectrogram_samples_explained"
thumb_dir = os.path.join(base, 'thumbnails')
os.makedirs(thumb_dir, exist_ok=True)
manifest = os.path.join(base, 'manifest_complete.csv')

rows = []
with open(manifest, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

failures = []
for row in rows:
    fn = row['filename']
    if fn.startswith('normal_'):
        src = os.path.join(base, 'normal_operations', fn)
    else:
        src = os.path.join(base, 'anomalies', fn)
    dst = os.path.join(thumb_dir, fn)
    try:
        with Image.open(src) as im:
            im = im.convert('RGB')
            im.thumbnail((240,240))
            im.save(dst, format='PNG')
    except Exception as e:
        print(f'Failed to thumbnail {src}: {e}')
        failures.append((src, str(e)))

# regenerate index.html
html_parts = []
html_parts.append('<!doctype html>')
html_parts.append('<html>')
html_parts.append('<head>')
html_parts.append("<meta charset='utf-8'>")
html_parts.append('<title>Spectrogram Samples - Index</title>')
html_parts.append('<style>')
html_parts.append('body{font-family:Arial,Helvetica,sans-serif;margin:20px}')
html_parts.append('.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));grid-gap:16px}')
html_parts.append('.card{border:1px solid #ddd;padding:8px;border-radius:6px;text-align:center}')
html_parts.append('.card img{max-width:220px;height:auto;border-radius:4px}')
html_parts.append('.label{font-weight:bold;margin-top:6px}')
html_parts.append('.desc{font-size:0.9em;color:#444;margin-top:4px}')
html_parts.append('a{color:inherit;text-decoration:none}')
html_parts.append('</style>')
html_parts.append('</head>')
html_parts.append('<body>')
html_parts.append('<h1>Spectrogram Samples - Index</h1>')
html_parts.append('<p>Click a thumbnail to open full-size image.</p>')
html_parts.append('<div class="grid">')

for row in rows:
    fn = row['filename']
    lab = row['label']
    desc = row['short_description']
    if fn.startswith('normal_'):
        fullpath = f"normal_operations/{fn}"
    else:
        fullpath = f"anomalies/{fn}"
    thumbpath = f"thumbnails/{fn}"
    card = f"<div class='card'><a href='{fullpath}' target='_blank'><img src='{thumbpath}' alt='{fn}'></a><div class='label'>{fn} - {lab}</div><div class='desc'>{desc}</div></div>"
    html_parts.append(card)

html_parts.append('</div>')
html_parts.append('</body>')
html_parts.append('</html>')

index_path = os.path.join(base, 'index.html')
with open(index_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(html_parts))

print(f'Thumbnails regenerated: {len(rows)-len(failures)} succeeded, {len(failures)} failed')
if failures:
    print('Failures:')
    for src, err in failures:
        print(src + ' -> ' + err)
