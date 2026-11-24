"""Generate the one-click M6 notebooks for Colab and Kaggle from one source, so both run the same protocol.

  python scripts/make_m6_notebook.py                      # writes both notebooks/m6_full_runs_{colab,kaggle}.ipynb
  python scripts/make_m6_notebook.py colab OUT.ipynb      # one platform

Only storage, bundle delivery and download differ between platforms. Model, decoding, training, judge and
checks are shared code.
"""
import json
import sys
from pathlib import Path


def build(platform: str) -> dict:
    cells = []

    def md(s):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip()})

    def code(s):
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip()})

    steps = """
What it does, in order (every finished step is saved and skipped next time):
0. **Preflight** (about 10 minutes, once): one tiny training update, llama.cpp serving the base model and a LoRA
   adapter, and the judge answering one request, so GPU problems show up before the long steps.
1. Builds the A/B/A′ training sets from the AI-reviewed data and checks the token budgets.
2. **Manipulation check (dev only):** trains A at 3 epochs and compares A with Base on `endorse` on the
   dev-exposed `conduit` cases. If A is not lower, retrains at 5 epochs (once). The chosen dose is saved.
3. Trains B and A′ at that dose.
4. Generates test transcripts for Base, P1, P2, A, B, A′ on all three suites (llama.cpp, f16 GGUF, LoRA applied at run time, same decoding).
5. Judges them with R4 (gpt-oss-20b via llama.cpp, built for the T4), then applies P-parse.
6. Runs the contact-number check and packages everything for download.

Help and helpfulness are graded afterwards on the laptop (R7), not here. Protocol: `docs/protocol-freeze.md`.
"""

    if platform == "colab":
        md("""
# M6: full runs on Colab (one click)

**Runtime > Change runtime type > T4 GPU**, then **Runtime > Run all**. Allow Google Drive (tick every
permission box) and upload `dist/colab_bundle.zip` when asked. Progress is saved to Google Drive, so if Colab
disconnects, reconnect and **Run all** again. At the end a zip downloads; put it in the repo's `runs/`.
""" + steps)
        code("""
from google.colab import drive
try:
    drive.mount('/content/drive')
except ValueError as e:
    raise RuntimeError('Google Drive did not connect. Run this cell again; in the sign-in popup pick your '
                       'account, tick EVERY permission box (or "Select all"), then Continue. Allow pop-ups '
                       'for colab.research.google.com if no popup appears.') from e
import os, time
T0 = time.time()
BASE = '/content/drive/MyDrive/selective-grounding'   # saved progress (Google Drive)
WORK = '/content/work'                                # unpacked code bundle
LOCAL = '/content'                                    # fast local disk: judge model, scratch, logs
DEADLINE_HOURS = None                                 # no budget: reconnect and Run all to continue
os.makedirs(BASE, exist_ok=True)
!nvidia-smi --query-gpu=name,memory.total --format=csv
""")
    else:
        md("""
# M6: full runs on Kaggle (one click, runs in the background)

**First time:**
1. kaggle.com > **Create > New Notebook**, then **File > Import Notebook** and choose
   `notebooks/m6_full_runs_kaggle.ipynb`.
2. Right-hand panel, **Session options**: Accelerator **GPU T4 x2**, Internet **On** (needs a phone-verified account).
3. **Add Input > Upload** (new dataset, private): upload `dist/colab_bundle.zip`. Any name is fine.
4. **Save Version > Save & Run All (Commit)**. It runs in the background; you can close the browser.

**When the version finishes:** open it, go to **Output**, and download `m6_outputs_e*.zip` into the repo's `runs/`.
If instead it says *TIME BUDGET REACHED* (Kaggle stops runs at 12 hours), continue: **Edit**, then **Add Input >
Your Work** and add this notebook's latest output, then **Save & Run All** again. Finished steps are skipped.
""" + steps)
        code("""
import glob, json, os, shutil, time
T0 = time.time()
BASE = '/kaggle/working/state'    # saved progress: kept as this version's output
WORK = '/tmp/work'                # unpacked code bundle
LOCAL = '/tmp/m6'                 # judge model, scratch, logs (not kept)
DEADLINE_HOURS = 10.5             # stop starting new steps after this, so the output is saved before Kaggle's 12 h limit
os.environ['CUDA_VISIBLE_DEVICES'] = '0'   # one T4, the same hardware as the Colab protocol
os.makedirs(LOCAL, exist_ok=True)
if not os.path.exists(f'{BASE}/M6_STATE'):
    # Resume from a previous version's output if it was added as an input (newest wins).
    prev = []
    for m in glob.glob('/kaggle/input/**/M6_STATE', recursive=True):
        try: prev.append((json.load(open(m)).get('updated', 0), os.path.dirname(m)))
        except Exception: pass
    if prev:
        src = max(prev)[1]
        shutil.copytree(src, BASE, dirs_exist_ok=True)
        print('resumed from', src)
    else:
        os.makedirs(BASE, exist_ok=True)
        print('fresh start (no previous output attached)')
    json.dump({'updated': time.time()}, open(f'{BASE}/M6_STATE', 'w'))
!nvidia-smi --query-gpu=name,memory.total --format=csv
""")

    code("""
MODEL = 'Qwen/Qwen3.5-4B'
SAMPLES = 1            # decoding samples per condition
MAX_TOKENS = 2048
TARGET_SLOTS = 4       # llama.cpp parallel slots for the target model
TARGET_CTX = 98304     # shared KV pool across slots (-kvu); longest dev conversation was ~11k tokens
WORKERS = TARGET_SLOTS # conversations in flight
GRAD_ACCUM = 4
DOSES = [3, 5]         # epochs: protocol dose, then the single pre-registered fallback
DEV_CASES = ['conduit_explicit', 'conduit_implicit']   # dev-exposed family, used only for the manipulation check
CONDITIONS = {  # label: (trained adapter or None, system prompt file or None)
    'base': (None, None), 'p1': (None, 'prompts/grounding.txt'), 'p2': (None, 'prompts/anti_sycophancy_reflection.txt'),
    'A': ('A', None), 'B': ('B', None), 'A_prime': ('A_prime', None)}
LLAMA_TAG = 'b12633'   # llama.cpp release: serves both the target (f16 GGUF + LoRA) and the judge
SUITES = ['psychosis-bench', 'grounded', 'grounded-risk']
JUDGE_SHA256 = '27cd6c432c7672cb812a92f611cf3ba7bbc35928262bb1e1253ff4ee6ae35901'
JUDGE_URL = ('https://huggingface.co/ggml-org/gpt-oss-20b-GGUF/resolve/'
             '41e8a94d15d62070b36cf4553a0250717ecf46b8/gpt-oss-20b-MXFP4.gguf')
""")

    # Keep the platform's own torch (built for its GPU driver, with a matching torchvision); pin the pure-Python
    # libraries to the versions that trained on Kaggle's T4.
    code("""
!pip -q install transformers==5.20.2 peft==0.21.2 accelerate pytest sentencepiece
# Unused here; peft refuses to create LoRA layers if an old version of either is preinstalled (Kaggle ships torchao 0.10).
!pip -q uninstall -y torchao gptqmodel
import torch, transformers, peft
print(torch.__version__, transformers.__version__, peft.__version__, 'CUDA OK' if torch.cuda.is_available() else 'NO CUDA')
assert torch.cuda.is_available(), 'No GPU: set the accelerator to a T4 GPU'
""")

    if platform == "colab":
        code("""
import os, shutil, subprocess, sys
if not os.path.exists(f'{WORK}/src'):
    from google.colab import files
    up = files.upload()  # choose dist/colab_bundle.zip (Colab may rename it, e.g. 'colab_bundle (1).zip')
    name = next(n for n in up if n.endswith('.zip'))
    with open(f'{LOCAL}/colab_bundle.zip', 'wb') as f:
        f.write(up[name])
    shutil.rmtree(WORK, ignore_errors=True)  # never mix with files from an older upload
    os.makedirs(WORK)
    subprocess.run(['unzip', '-q', f'{LOCAL}/colab_bundle.zip', '-d', WORK], check=True)
""")
    else:
        code("""
import os, shutil, subprocess, sys
if not os.path.exists(f'{WORK}/src'):
    # Kaggle usually unpacks an uploaded zip; handle both the unpacked folder and the zip itself.
    hits = glob.glob('/kaggle/input/**/src/psyeval/__init__.py', recursive=True)
    zips = glob.glob('/kaggle/input/**/colab_bundle*.zip', recursive=True)
    shutil.rmtree(WORK, ignore_errors=True)
    if hits:
        shutil.copytree(os.path.dirname(os.path.dirname(os.path.dirname(hits[0]))), WORK)
    elif zips:
        os.makedirs(WORK)
        subprocess.run(['unzip', '-q', zips[0], '-d', WORK], check=True)
    else:
        raise RuntimeError('Bundle not found: Add Input > Upload a dataset containing dist/colab_bundle.zip')
""")

    code("""
# Shared from here on. Code checks first.
os.chdir(WORK)
if not os.path.isdir('third_party/psychosis-bench'):
    subprocess.run('git clone -q https://github.com/w-is-h/psychosis-bench third_party/psychosis-bench && cd third_party/psychosis-bench '
                   '&& git checkout -q 73966f95be2e40f1ceee76dfe08eebe939ad6c21', shell=True, check=True)
r = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-rf'], capture_output=True, text=True)
print(r.stdout[-4000:], r.stderr[-2000:])
if r.returncode:
    raise RuntimeError('tests failed: see the lines above, and send them to Claude')
""")

    code("""
# Helpers. Every output is written to <out>.tmp and renamed only when the step succeeds,
# so an interrupted step is redone from scratch and a finished one is skipped.
import glob, hashlib, json, shutil, subprocess, sys, time, urllib.request
ENV = {**os.environ, 'PYTHONPATH': 'src'}
DEADLINE = T0 + DEADLINE_HOURS * 3600 if DEADLINE_HOURS else float('inf')
STOPPED = False
SERVERS = []

class OutOfTime(Exception):
    pass

def sh(cmd, quiet=False, env=None, **kw):
    # Notebooks do not show a child process's own output, so stream it into the cell; on failure,
    # repeat the last lines in the error so the real cause is visible.
    cmd = [str(c) for c in cmd]
    env = env or ENV
    print('$', ' '.join(cmd)[:240], flush=True)
    if kw:  # callers that capture output (stdout is parsed); stderr still shown on failure
        r = subprocess.run(cmd, env=env, **kw)
        if r.returncode:
            raise RuntimeError(f'exit {r.returncode}: ' + str(r.stderr)[-3000:])
        return r
    tail = []
    with subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as p:
        for line in p.stdout:
            if not quiet:
                print(line, end='', flush=True)
            tail = (tail + [line])[-40:]
    if p.returncode:
        raise RuntimeError(f'exit {p.returncode}. Last lines:\\n' + ''.join(tail))
    return p

def mark_progress():
    json.dump({'updated': time.time()}, open(f'{BASE}/M6_STATE', 'w'))

def atomic(out, build):
    if os.path.isdir(out):
        print('skip (done):', out); return
    if time.time() > DEADLINE:
        raise OutOfTime()
    tmp = out + '.tmp'
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    build(tmp)
    os.rename(tmp, out)
    mark_progress()

def step(fn):
    # Runs one notebook section. If the time budget runs out, later sections are skipped cleanly so the
    # progress so far is saved; servers are always stopped.
    global STOPPED
    if STOPPED:
        print('skipped: time budget reached in this session'); return
    try:
        fn()
    except OutOfTime:
        STOPPED = True
        print(f'TIME BUDGET REACHED ({DEADLINE_HOURS} h). Progress is saved; run again with this output attached.')
    finally:
        for p in SERVERS:
            if p.poll() is None:
                stop(p)
        SERVERS.clear()

def wait_up(port, proc, log):
    for _ in range(240):
        if proc.poll() is not None:
            raise RuntimeError(f'server on :{port} exited. Last lines of {log}:\\n' + ''.join(open(log).readlines()[-40:]))
        try:
            urllib.request.urlopen(f'http://127.0.0.1:{port}/health'); print(f'server :{port} up'); return
        except Exception:
            time.sleep(10)
    raise RuntimeError(f'server on :{port} did not start. Last lines of {log}:\\n' + ''.join(open(log).readlines()[-40:]))

def stop(proc):
    proc.terminate()
    try: proc.wait(timeout=180)
    except subprocess.TimeoutExpired: proc.kill()
    time.sleep(10)

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            h.update(chunk)
    return h.hexdigest()

def lora_to_gguf(adapter_dir, out_file):
    # PEFT adapter -> GGUF LoRA with llama.cpp's own converter (same tag as the server).
    sh([sys.executable, f'{LLAMA_SRC}/convert_lora_to_gguf.py', adapter_dir, '--base', HF_DIR, '--outtype', 'f16',
        '--outfile', out_file + '.part'], quiet=True, env={**ENV, 'PYTHONPATH': f'{LLAMA_SRC}/gguf-py'})
    os.rename(out_file + '.part', out_file)

TARGET = {'lora': None}   # what the running target server has loaded

def adapter_gguf(label, epochs):
    adapter = CONDITIONS[label][0]
    return f'{BASE}/adapters/{adapter}-e{epochs}/adapter.gguf' if adapter else None

def start_target(lora=None):
    # The target model as f16 GGUF on llama.cpp; a condition's LoRA (GGUF) is applied at scale 1 for its whole run.
    cmd = [LLAMA, '-m', TARGET_GGUF, '--alias', 'target', '--port', '8000', '-ngl', '99', '-c', TARGET_CTX,
           '-np', TARGET_SLOTS, '-kvu', '--seed', '1234', '--reasoning', 'off']
    if lora:
        cmd += ['--lora', lora]
    p = subprocess.Popen([str(c) for c in cmd], stdout=open(f'{LOCAL}/target.log', 'w'), stderr=subprocess.STDOUT)
    SERVERS.append(p)
    wait_up(8000, p, f'{LOCAL}/target.log')
    TARGET.update(lora=lora, cmd=[str(c) for c in cmd[1:]], lora_sha256=sha256_file(lora) if lora else None)
    return p

def start_judge():
    p = subprocess.Popen([LLAMA, '-m', JUDGE_FILE, '--alias', 'gpt-oss-20b', '--port', '8082', '-ngl', '99', '-c', '65536',
                          '-np', '4', '-kvu', '--seed', '1234', '--reasoning-effort', 'low'],
                         stdout=open(f'{LOCAL}/judge.log', 'w'), stderr=subprocess.STDOUT)
    SERVERS.append(p)
    wait_up(8082, p, f'{LOCAL}/judge.log'); return p

def train(cond, epochs):
    # Train on local disk (a network mount is slow and can lag on delete/create), convert the adapter to GGUF, then copy.
    out = f'{BASE}/adapters/{cond}-e{epochs}'
    local = f'{LOCAL}/scratch/{cond}-e{epochs}'
    def build(tmp):
        shutil.rmtree(local, ignore_errors=True)
        os.makedirs(f'{LOCAL}/scratch', exist_ok=True)
        sh([sys.executable, 'scripts/train_lora.py', '--model', MODEL, '--data', f'data/conditions/{cond}.jsonl',
            '--epochs', epochs, '--grad-accum', GRAD_ACCUM, '--dtype', 'float16', '--grad-checkpointing', '--out', local])
        lora_to_gguf(f'{local}/adapter', f'{local}/adapter.gguf')
        shutil.copytree(local, tmp)
    atomic(out, build)
    return out

def generate(label, suite, out, lora, split='test', cases=None, seed=1000):
    sp = CONDITIONS[label][1]
    assert TARGET['lora'] == lora, f'target server has {TARGET["lora"]}, but {label} needs {lora}'
    def build(tmp):
        provenance = {'engine': f'llama.cpp {LLAMA_TAG}', 'target_gguf': TARGET_INFO, 'server_args': TARGET['cmd'],
                      'lora_gguf_sha256': TARGET['lora_sha256']}
        cmd = [sys.executable, '-m', 'psyeval', 'run', '--suite', suite, '--split', split, '--label', label,
               '--target-base-url', 'http://127.0.0.1:8000/v1', '--target-model', 'target', '--temperature', '0.7',
               '--max-tokens', MAX_TOKENS, '--seed', seed, '--timeout', '900', '--no-judge', '--workers', WORKERS, '--out', tmp,
               # Non-thinking mode, matching training (enable_thinking=False) and the local dev runs (--reasoning off).
               '--target-extra-body', json.dumps({'chat_template_kwargs': {'enable_thinking': False}}),
               '--manifest-extra', json.dumps(provenance)]
        if sp: cmd += ['--system-prompt-file', sp]
        if cases: cmd += ['--cases', ','.join(cases)]
        sh(cmd)
    atomic(out, build)

def generate_conditions(labels, suites, out_dir, epochs, **kw):
    # One target server per adapter (base, P1 and P2 share the plain model), every suite for each condition.
    # Output folders are <out_dir>/<label>-<suite>.
    groups = {}
    for label in labels:
        groups.setdefault(adapter_gguf(label, epochs), []).append(label)
    for lora, group in groups.items():
        outs = [(label, suite, f'{out_dir}/{label}-{suite}') for label in group for suite in suites]
        if all(os.path.isdir(o) for _, _, o in outs):
            for _, _, o in outs: print('skip (done):', o)
            continue
        if time.time() > DEADLINE:
            raise OutOfTime()
        v = start_target(lora)
        for label, suite, o in outs:
            generate(label, suite, o, lora, **kw)
        stop(v)

def judge(gen, out):
    atomic(out, lambda tmp: sh([sys.executable, '-m', 'psyeval', 'judge', gen, '--judge-base-url', 'http://127.0.0.1:8082/v1',
                                '--judge-model', 'gpt-oss-20b', '--judge-style', 'reasoned', '--judge-max-tokens', '2048',
                                '--judge-timeout', '1800', '--workers', '4', '--out', tmp]))
    shutil.rmtree(out + '-reparsed', ignore_errors=True)
    sh([sys.executable, 'scripts/reparse_runs.py', out])  # P-parse, no judge calls
    return out + '-reparsed'

def endorse_mean(run):
    s = json.loads(sh([sys.executable, '-m', 'psyeval', 'summarize', run], capture_output=True, text=True).stdout)
    return s['metrics']['endorse']['complete_case']['mean_of_conversation_means']

def chat(port, model, text, extra=None, max_tokens=64, thinking_allowed=True):
    body = {'model': model, 'messages': [{'role': 'user', 'content': text}], 'max_tokens': max_tokens,
            'temperature': 0, **(extra or {})}
    req = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions', json.dumps(body).encode(),
                                 {'Content-Type': 'application/json'})
    out = json.load(urllib.request.urlopen(req, timeout=600))['choices'][0]['message']
    content, reasoning = out.get('content') or '', out.get('reasoning_content') or ''
    if not thinking_allowed:  # the target must answer in non-thinking mode, as it was trained
        assert not reasoning.strip() and '<think>' not in content, f'thinking mode is on for {model}: {reasoning[:200]}'
    reply = content + reasoning
    assert reply.strip(), f'empty reply from {model} on :{port}'
    return reply
""")

    code("""
# llama.cpp from the pinned release tag: its server (built for the T4, compute capability 7.5) runs both the
# target model and the judge, and its converters turn the model and each LoRA adapter into GGUF. The server
# binary is cached with the saved progress; models go on local disk.
os.makedirs(f'{BASE}/judge', exist_ok=True); os.makedirs(f'{LOCAL}/judge', exist_ok=True)
LLAMA_SRC = f'{LOCAL}/llama.cpp'
if not os.path.exists(f'{LLAMA_SRC}/convert_hf_to_gguf.py'):
    shutil.rmtree(LLAMA_SRC, ignore_errors=True)
    sh(['git', '-c', 'advice.detachedHead=false', 'clone', '-q', '--depth', '1', '--branch', LLAMA_TAG,
        'https://github.com/ggml-org/llama.cpp', LLAMA_SRC])
LLAMA_CACHE = f'{BASE}/judge/llama-server'
if not os.path.exists(LLAMA_CACHE):
    if not shutil.which('cmake'):
        sh([sys.executable, '-m', 'pip', 'install', '-q', 'cmake'])
    nvcc = shutil.which('nvcc') or '/usr/local/cuda/bin/nvcc'
    src = LLAMA_SRC
    # Some images (e.g. Kaggle, CUDA 13) ship the toolkit without the libcuda link stub, so CMake cannot find
    # CUDA::cuda_driver. Point it at the stub if present, otherwise at the installed driver library.
    libcuda = [p for pat in ('/usr/local/cuda/lib64/stubs/libcuda.so', '/usr/local/cuda/targets/*/lib/stubs/libcuda.so',
                             '/usr/lib/x86_64-linux-gnu/libcuda.so', '/usr/lib/x86_64-linux-gnu/libcuda.so.1',
                             '/usr/local/nvidia/lib64/libcuda.so*', '/usr/lib64/libcuda.so*')
               for p in sorted(glob.glob(pat))]
    print('libcuda for linking:', libcuda[0] if libcuda else 'not found (letting CMake search)')
    print('building llama-server (10-25 minutes)...', flush=True)
    sh(['cmake', '-S', src, '-B', f'{src}/build', '-DGGML_CUDA=ON', '-DCMAKE_CUDA_ARCHITECTURES=75', f'-DCMAKE_CUDA_COMPILER={nvcc}',
        '-DBUILD_SHARED_LIBS=OFF', '-DLLAMA_CURL=OFF', *([f'-DCUDA_cuda_driver_LIBRARY={libcuda[0]}'] if libcuda else [])], quiet=True)
    sh(['cmake', '--build', f'{src}/build', '-j', '4', '--target', 'llama-server'], quiet=True)
    shutil.copy(f'{src}/build/bin/llama-server', LLAMA_CACHE)
LLAMA = f'{LOCAL}/judge/llama-server'   # run from local disk
shutil.copy(LLAMA_CACHE, LLAMA)
os.chmod(LLAMA, 0o755)
print('llama-server ready')
""")

    code("""
# Judge: the same gpt-oss-20b file as the local dev runs, from a pinned Hugging Face commit, checksum-verified.
JUDGE_FILE = f'{LOCAL}/judge/gpt-oss-20b-MXFP4.gguf'
if not os.path.exists(JUDGE_FILE):
    print('downloading the judge model (12.1 GB, a few minutes)...', flush=True)
    sh(['curl', '-fsSL', '--retry', '8', '-C', '-', '-o', JUDGE_FILE + '.part', JUDGE_URL])
    os.rename(JUDGE_FILE + '.part', JUDGE_FILE)
assert sha256_file(JUDGE_FILE) == JUDGE_SHA256, 'judge file checksum mismatch: delete it and re-run'
print('judge ready')
""")

    code("""
# Target: Qwen3.5-4B converted to an f16 GGUF (no quantisation) with llama.cpp's converter, about 5 minutes.
from huggingface_hub import snapshot_download
HF_DIR = snapshot_download(MODEL, allow_patterns=['*.json', '*.safetensors', '*.jinja', '*.txt'])
TARGET_GGUF = f'{LOCAL}/target/qwen3.5-4b-f16.gguf'
if not os.path.exists(TARGET_GGUF):
    os.makedirs(f'{LOCAL}/target', exist_ok=True)
    print('converting the target model to f16 GGUF...', flush=True)
    sh([sys.executable, f'{LLAMA_SRC}/convert_hf_to_gguf.py', HF_DIR, '--outtype', 'f16', '--outfile', TARGET_GGUF + '.part'],
       quiet=True, env={**ENV, 'PYTHONPATH': f'{LLAMA_SRC}/gguf-py'})
    os.rename(TARGET_GGUF + '.part', TARGET_GGUF)
TARGET_INFO = {'hf_model': MODEL, 'hf_revision': os.path.basename(HF_DIR.rstrip('/')), 'dtype': 'f16',
               'sha256': sha256_file(TARGET_GGUF)}
json.dump(TARGET_INFO, open(f'{BASE}/target_gguf.json', 'w'), indent=2)
print('target ready:', TARGET_INFO)
""")

    md("## 0. Preflight (about 10 minutes, once)")

    code("""
PREFLIGHT = f'{BASE}/preflight_ok.json'

def _preflight():
    if os.path.exists(PREFLIGHT):
        print('skip (done): preflight'); return
    pf = f'{LOCAL}/scratch/preflight'
    shutil.rmtree(pf, ignore_errors=True); os.makedirs(f'{LOCAL}/scratch', exist_ok=True)
    sh([sys.executable, 'scripts/train_lora.py', '--model', MODEL, '--data', 'data/drafts/train_concerning.jsonl',
        '--limit', 4, '--grad-accum', 1, '--max-steps', 1, '--dtype', 'float16', '--grad-checkpointing', '--out', pf])
    lora_to_gguf(f'{pf}/adapter', f'{pf}/adapter.gguf')
    no_think = {'chat_template_kwargs': {'enable_thinking': False}}
    v = start_target()
    r1 = chat(8000, 'target', 'Say hello in five words.', no_think, thinking_allowed=False)
    stop(v)
    v = start_target(f'{pf}/adapter.gguf')
    r2 = chat(8000, 'target', 'Say hello in five words.', no_think, thinking_allowed=False)
    stop(v)
    print('base:', r1[:120]); print('adapter:', r2[:120])
    j = start_judge()
    r3 = chat(8082, 'gpt-oss-20b', 'Reply with the single word OK.', max_tokens=256)
    print('judge:', r3[:120])
    stop(j)
    json.dump({'ok': True, 'base': r1[:200], 'adapter': r2[:200], 'judge': r3[:200]}, open(PREFLIGHT, 'w'), indent=2)
    print('PREFLIGHT PASSED')

step(_preflight)
""")

    md("## 1. Training sets (AI-reviewed exploratory data)")

    code("""
sh([sys.executable, 'scripts/build_conditions.py', '--out', 'data/conditions', '--allow-ai-reviewed', '--tokenizer', MODEL])
manifest = json.load(open('data/conditions/manifest.json'))
assert not manifest['missing_strata_for_B']
assert manifest['target_token_relative_difference_A_B'] <= 0.05, 'Review token budgets before training'
assert all(c['n'] > 0 and c['target_check_failures'] == 0 for c in manifest['conditions'].values())
shutil.copy('data/conditions/manifest.json', f'{BASE}/conditions_manifest.json')
print({k: (v['n'], v['target_tokens']) for k, v in manifest['conditions'].items()})
""")

    md("## 2. Manipulation check and dose (dev only, rule fixed in advance)")

    code("""
DOSE_FILE = f'{BASE}/dose.json'

def _dose():
    if os.path.exists(DOSE_FILE):
        return
    log = []
    for ep in DOSES:
        train('A', ep)
        gen = f'{BASE}/runs/check-e{ep}/gen'
        generate_conditions(['base', 'A'], ['psychosis-bench'], gen, ep, split='dev', cases=DEV_CASES)
        j = start_judge()
        res = {label: endorse_mean(judge(f'{gen}/{label}-psychosis-bench', f'{BASE}/runs/check-e{ep}/judged/{label}-psychosis-bench'))
               for label in ['base', 'A']}
        stop(j)
        log.append({'epochs': ep, 'endorse_base': res['base'], 'endorse_A': res['A']})
        print(log[-1])
        if res['A'] is not None and res['base'] is not None and res['A'] < res['base']:
            break
    last = log[-1]
    dose = {'epochs': last['epochs'], 'passed': last['endorse_A'] is not None and last['endorse_base'] is not None
            and last['endorse_A'] < last['endorse_base'], 'log': log}
    json.dump(dose, open(DOSE_FILE, 'w'), indent=2)
    mark_progress()

step(_dose)
if os.path.exists(DOSE_FILE):
    DOSE = json.load(open(DOSE_FILE))['epochs']
    RUNS = f'{BASE}/runs/test-e{DOSE}'
    print('dose:', json.load(open(DOSE_FILE)))
""")

    md("## 3. Train all adapters at the chosen dose")

    code("""
def _train_all():
    for c in ['A', 'B', 'A_prime']:
        train(c, DOSE)
        m = json.load(open(f'{BASE}/adapters/{c}-e{DOSE}/manifest.json'))
        print(c, {k: m.get(k) for k in ('optimizer_updates', 'target_tokens_seen', 'epochs_equivalent', 'wall_time_s')})

step(_train_all)
""")

    md("## 4. Generate test transcripts (no judge)")

    code("""
def _generate_all():
    for s in range(SAMPLES):
        generate_conditions(list(CONDITIONS), SUITES, f'{RUNS}/gen-s{s}', DOSE, seed=1000 + s)

step(_generate_all)
""")

    md("## 5. Judge with R4, then P-parse")

    code("""
def _judge_all():
    j = start_judge()
    for gen in sorted(glob.glob(f'{RUNS}/gen-s*/*')):
        if not gen.endswith('.tmp'):
            judge(gen, gen.replace('/gen-', '/judged-'))
    stop(j)

step(_judge_all)
""")

    md("## 6. Contact check, quick summaries, and download")

    code("""
def _report():
    gens = [g for g in sorted(glob.glob(f'{RUNS}/gen-s*/*')) if not g.endswith('.tmp')]
    cc = sh([sys.executable, 'scripts/contact_check.py', *gens], capture_output=True, text=True).stdout
    open(f'{RUNS}/contact_check.json', 'w').write(cc)
    for r in json.loads(cc):
        print(os.path.basename(r['run']), 'unlisted-number replies:', r['replies_with_unlisted_number'], '/', r['replies'])
    for d in sorted(glob.glob(f'{RUNS}/judged-s0/*-reparsed')):
        X = ['--exclude-families', 'conduit'] if 'psychosis-bench' in d else []
        s = json.loads(sh([sys.executable, '-m', 'psyeval', 'summarize', d, *X], capture_output=True, text=True).stdout)
        print(os.path.basename(d), {k: round(v['complete_case']['mean_of_conversation_means'], 2)
                                    for k, v in s['metrics'].items() if v['complete_case']['mean_of_conversation_means'] is not None})

step(_report)
""")

    download = ("from google.colab import files\n    files.download(zpath)" if platform == "colab" else
                "print('Done. Download it from this version\\'s Output tab.')")
    zdir = "LOCAL" if platform == "colab" else "'/kaggle/working'"
    code(f"""
import zipfile
if STOPPED:
    print('Not packaged: the run is not finished. Run again to continue.')
else:
    zpath = f'{{{zdir}}}/m6_outputs_e{{DOSE}}.zip'
    with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in [DOSE_FILE, f'{{BASE}}/conditions_manifest.json', f'{{BASE}}/preflight_ok.json',
                  *glob.glob(f'{{BASE}}/adapters/*/manifest.json')]:
            if os.path.exists(f):
                z.write(f, os.path.relpath(f, BASE))
        for f in glob.glob(f'{{BASE}}/runs/**/*', recursive=True):
            if os.path.isfile(f) and '.tmp' not in f:
                z.write(f, os.path.relpath(f, BASE))
    print(zpath, round(os.path.getsize(zpath) / 1e6, 1), 'MB')
    {download}
""")

    meta = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"}}
    if platform == "colab":
        meta.update({"accelerator": "GPU", "colab": {"provenance": []}})
    return {"cells": cells, "metadata": meta, "nbformat": 4, "nbformat_minor": 0}


def write(platform: str, out: str) -> None:
    nb = build(platform)
    Path(out).write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"{platform}: {out} ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        write(sys.argv[1], sys.argv[2])
    else:
        write("colab", "notebooks/m6_full_runs_colab.ipynb")
        write("kaggle", "notebooks/m6_full_runs_kaggle.ipynb")
