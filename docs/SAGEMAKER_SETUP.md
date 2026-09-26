# SageMaker Setup Guide — AMLC 2026

How to build the same environment we're running on: **8 vCPU, 32 GiB RAM, 100 GB disk**,
dataset loaded, code cloned, smoke test passing.

**Time: ~30 minutes**, most of it waiting for AWS.
Written from an actual setup — every error in the troubleshooting section is one we hit.

> Replace these with your own values throughout:
> - `<DOMAIN_ID>` — looks like `d-xxxxxxxxxxxx`
> - `<SPACE_NAME>` — looks like `quickstart-default-xxxxxx`
> - `<YOUR_BUCKET>` — your SageMaker bucket, `sagemaker-us-east-1-<your-account-id>`
> - Region below is `us-east-1`. If you use another, change it **everywhere** — the
>   image ARN in Part 4 is region-specific.

---

## Part 0 — What you're building

| | value |
|---|---|
| Instance | `ml.m7i.2xlarge` — 8 vCPU, 32 GiB, Sapphire Rapids |
| Storage | 100 GB EBS (persists when the app is stopped) |
| Cost | ~$0.40–0.60/hr compute + ~$10/mo storage prorated |
| Region | us-east-1 |

**Why this instance:** sustained performance, not burstable. Avoid the `ml.t3.*` family for
long jobs — they run on CPU credits and throttle to ~40% baseline once those run out, which
is exactly wrong for a multi-hour batch run.

**Why 100 GB:** the dataset is 2.4 GB, and `candidate_pairs.tsv` at K=150 is ~12.6 GB as TSV.
The default 5 GB space cannot hold the pipeline.

---

## Part 1 — Prerequisites

1. **An AWS account** with credits applied.
2. **AWS CLI v2** on your laptop:
   ```bash
   aws --version        # need 2.x
   aws configure        # access key, secret, region us-east-1
   aws sts get-caller-identity   # should print your account id
   ```
   If `aws` is missing: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html

---

## Part 2 — Create the SageMaker domain (console)

Easiest in the browser; the CLI equivalent needs IAM roles and VPC config you don't want to
hand-write.

1. Go to **SageMaker AI** in the AWS console, region **us-east-1**.
2. Left sidebar → **Domains** → **Create domain**.
3. Choose **Quick setup**. Accept the defaults; it creates the execution role for you.
4. Wait ~5 minutes for `InService`.

This gives you a domain, a user profile, and a default JupyterLab space on `ml.t3.medium`
(2 vCPU, 4 GiB, 5 GB disk) — too small, which Part 4 fixes.

Get your IDs:

```bash
export AWS_PAGER=""
aws sagemaker list-domains --region us-east-1 \
  --query 'Domains[].{Name:DomainName,Id:DomainId,Status:Status}' --output table

aws sagemaker list-spaces --region us-east-1 \
  --query 'Spaces[].{Space:SpaceName,Status:Status}' --output table
```

Note the `DomainId` and `SpaceName` — you need them for every command below.

---

## Part 3 — Check your quotas ⚠️ DO THIS BEFORE ANYTHING ELSE

**This is the step that will waste your time if you skip it.** New AWS accounts have a
quota of **0** for most instance types. Credits do not change quotas. A quota increase
request can take hours to days.

```bash
aws service-quotas list-service-quotas --service-code sagemaker --region us-east-1 \
  --query "Quotas[?contains(QuotaName,'JupyterLab')].[Value,QuotaName]" --output text \
  | awk '$1>0' | sort -rn
```

This takes ~60 seconds. Anything not listed with a value ≥ 1 **cannot be launched.**

On our account the usable ones were:

| Instance | vCPU | RAM | Verdict |
|---|---|---|---|
| **ml.m7i.2xlarge** | 8 | 32 GB | ✅ **use this** |
| ml.c6i.2xlarge | 8 | 16 GB | ok, less RAM |
| ml.r7i.xlarge | 4 | 32 GB | ok, fewer cores |
| ml.t3.2xlarge | 8 | 32 GB | ⚠️ burstable — throttles |
| ml.c5.4xlarge | 16 | 32 GB | ❌ quota 0 |
| ml.m5.4xlarge | 16 | 64 GB | ❌ quota 0 |

If your quota list is different, pick the best available with **≥16 GB RAM and ≥4 vCPU**,
preferring `m7i` / `c6i` / `r7i` over `t3`.

**If everything is 0:** request an increase immediately (Service Quotas console → search
"JupyterLab" → select the type → Request increase). Then work locally until it lands.

---

## Part 4 — Upgrade the instance and storage

An app's instance type is fixed at creation, so you delete and recreate it. **The EBS
volume and everything on it survives** — only the compute is replaced.

```bash
export AWS_PAGER=""
DOMAIN=<DOMAIN_ID>
SPACE=<SPACE_NAME>
REGION=us-east-1

# 1. Delete the small app (EBS survives)
aws sagemaker delete-app --domain-id $DOMAIN --space-name $SPACE \
  --app-type JupyterLab --app-name default --region $REGION

# wait for it to go (~30s)
while aws sagemaker describe-app --domain-id $DOMAIN --space-name $SPACE \
      --app-type JupyterLab --app-name default --region $REGION \
      --query 'Status' --output text 2>/dev/null | grep -q Deleting; do
  echo "  deleting..."; sleep 10
done

# 2. Grow storage 5 GB -> 100 GB.  CAN GROW, CANNOT SHRINK.
aws sagemaker update-space --domain-id $DOMAIN --space-name $SPACE --region $REGION \
  --space-settings 'SpaceStorageSettings={EbsStorageSettings={EbsVolumeSizeInGb=100}}'

# 3. Recreate on the bigger instance.  The image ARN is REQUIRED and region-specific.
aws sagemaker create-app --domain-id $DOMAIN --space-name $SPACE \
  --app-type JupyterLab --app-name default --region $REGION \
  --resource-spec InstanceType=ml.m7i.2xlarge,SageMakerImageArn=arn:aws:sagemaker:us-east-1:885854791233:image/sagemaker-distribution-cpu
```

Wait for `InService` — takes **3–5 minutes**:

```bash
while true; do
  S=$(aws sagemaker describe-app --domain-id $DOMAIN --space-name $SPACE \
      --app-type JupyterLab --app-name default --region $REGION \
      --query 'Status' --output text)
  echo "  $S"; [ "$S" = "InService" ] && break
  [ "$S" = "Failed" ] && { echo "FAILED"; break; }
  sleep 20
done
```

Open JupyterLab from the console (**Studio → JupyterLab → Open**) and verify in a terminal:

```bash
free -h      # ~30Gi total
nproc        # 8
df -h ~      # ~100G
```

---

## Part 5 — Get the dataset in

**Your home directory is `/home/sagemaker-user/`.** Not `~/SageMaker/`, not
`/home/ec2-user/` — those are SageMaker *Notebook Instance* paths and don't exist in Studio.

### The permission trap

Your SageMaker **execution role** is a different identity from your laptop's **IAM user**.
A bucket your laptop can read is usually *not* readable from the instance. You'll see:

```
An error occurred (AccessDenied) when calling the GetObject operation
```

### Fix: copy the dataset into your own bucket

Run this **on your laptop** (it has the credentials that can read the shared bucket):

```bash
aws s3 sync \
  s3://<SHARED_BUCKET>/Amazon-ML-dataset/student_resource/dataset/ \
  s3://<YOUR_BUCKET>/amlc/dataset/ \
  --copy-props none --exclude "*.DS_Store"
```

`--copy-props none` is required — without it the copy tries to carry object tags and fails
with `GetObjectTagging: Access Denied` if you only have object-read permission.

This is a **server-side** copy: the bytes move inside AWS at ~130 MiB/s and never touch your
home connection. 2.3 GB takes about 20 seconds.

Then **on the instance**:

```bash
cd ~/amazon-ml-hackathon-26
aws s3 sync s3://<YOUR_BUCKET>/amlc/dataset/ ./dataset/
du -sh dataset/ && ls dataset/train/ dataset/test/
```

Expect **2.3 GB, 7 files**. Same region, so this is also fast.

**No S3 copy of the dataset?** Upload from your laptop instead — slower, limited by your
upload speed:
```bash
aws s3 sync /path/to/student_resource/dataset/ s3://<YOUR_BUCKET>/amlc/dataset/
```

---

## Part 6 — Get the code in

```bash
cd ~
git clone https://github.com/yash-7575/amazon-ml-hackathon-26.git
cd amazon-ml-hackathon-26
```

**If the repo is private, this will fail.** GitHub disabled password authentication for git
in August 2021 — the prompt still says "Password" but your password will never work.

Options:
- **Make the repo public** (what we did — simplest)
- **Personal Access Token**: github.com → Settings → Developer settings → Personal access
  tokens → Fine-grained → Contents: read/write. Then
  `git clone https://USER:TOKEN@github.com/...` *(the token lands in your shell history and
  `.git/config` — acceptable on a scratch instance, not elsewhere)*
- **Skip git**: tar the code on your laptop, put it in S3, pull it down. Fast for a small repo.

### Point the code at the data

```bash
sed -i "s|/home/yash/Downloads/Amazon-ML-dataset/student_resource/dataset/|/home/sagemaker-user/amazon-ml-hackathon-26/dataset/|" blocking/config.py
grep -n data_root blocking/config.py     # verify it changed
```

### Dependencies

```bash
pip install -q Unidecode indic-transliteration
python -c "import numpy,scipy,sklearn;print(numpy.__version__,scipy.__version__,sklearn.__version__)"
```

numpy / scipy / scikit-learn ship with the SageMaker distribution image.

---

## Part 7 — Verify

```bash
PYTHONPATH=. python blocking/smoke_test.py india 2000 60000 60000
```

Takes ~45 seconds. You should see approximately:

```
recall P1+P2+P3 (union)  : 98.3%
  K=150  final recall 98.3%   cross-script 94%
peak RSS  ~4390 MiB
```

Small differences (±0.1pp) are sampling jitter, not a problem. If you're off by whole
percentage points, the dataset or config path is wrong.

---

## Part 8 — Cost control ⚠️

**The instance bills for every hour it's `InService`, whether or not you're using it.**

Stop it when you're done for the night:

```bash
aws sagemaker delete-app --domain-id $DOMAIN --space-name $SPACE \
  --app-type JupyterLab --app-name default --region $REGION
```

This deletes **only the compute**. Your 100 GB EBS and every file on it survives. Restart
with the same `create-app` command from Part 4 (image ARN included).

**Never delete the *space*** — that destroys the EBS volume and your work with it.

Check what's running and billing:

```bash
aws sagemaker list-apps --domain-id-equals $DOMAIN --region $REGION \
  --query 'Apps[?Status==`InService`].{App:AppName,Instance:ResourceSpec.InstanceType}' \
  --output table
```

EBS bills ~$0.10/GB-month regardless of whether the app runs — 100 GB ≈ $10/month, so
about $0.35/day. Trivial, but delete the space entirely when the competition ends.

---

## Troubleshooting — errors we actually hit

**`SageMaker Image ARN is required for App with type [JupyterLab]`**
`create-app` needs the image. Add to `--resource-spec`:
`SageMakerImageArn=arn:aws:sagemaker:us-east-1:885854791233:image/sagemaker-distribution-cpu`
That account id is AWS's, not yours, and the region must match yours.

**`ResourceLimitExceeded` on create-app**
Quota 0 for that instance type. Go back to Part 3 and pick one with quota ≥ 1.

**`AccessDenied` on `GetObject` when syncing the dataset**
Execution role can't read that bucket. Copy the data into your own bucket (Part 5).

**`AccessDenied` on `GetObjectTagging` during an S3→S3 copy**
Add `--copy-props none`.

**`Password authentication is not supported for Git operations`**
GitHub killed password auth in 2021. Public repo, PAT, or S3 (Part 6).

**Killed / OOM during a run**
Check `free -h`. If you're on `ml.t3.medium` (4 GiB), Part 4 wasn't applied — the app was
recreated but check it actually came back on the bigger type:
`aws sagemaker describe-app ... --query 'ResourceSpec.InstanceType'`

**Job slows down badly after ~20 minutes**
You're on a `t3` burstable instance and CPU credits ran out. Switch to `m7i`/`c6i`/`r7i`.

**`No space left on device`**
EBS still 5 GB. Verify:
`aws sagemaker describe-space ... --query 'SpaceSettings.SpaceStorageSettings'`
The update only takes effect after the app is recreated.

**Files vanished after stopping**
You deleted the *space*, not the *app*. Only `delete-app` is safe. Anything outside
`/home/sagemaker-user/` is ephemeral regardless — keep work there and back up to S3.

---

## Quick reference

```bash
export AWS_PAGER=""
DOMAIN=<DOMAIN_ID>; SPACE=<SPACE_NAME>; REGION=us-east-1
IMG=arn:aws:sagemaker:us-east-1:885854791233:image/sagemaker-distribution-cpu

# start
aws sagemaker create-app --domain-id $DOMAIN --space-name $SPACE \
  --app-type JupyterLab --app-name default --region $REGION \
  --resource-spec InstanceType=ml.m7i.2xlarge,SageMakerImageArn=$IMG

# stop (keeps all files)
aws sagemaker delete-app --domain-id $DOMAIN --space-name $SPACE \
  --app-type JupyterLab --app-name default --region $REGION

# status
aws sagemaker describe-app --domain-id $DOMAIN --space-name $SPACE \
  --app-type JupyterLab --app-name default --region $REGION \
  --query '{Status:Status,Instance:ResourceSpec.InstanceType}' --output table
```

On the instance: home is `/home/sagemaker-user/` · `free -h` · `nproc` · `df -h ~`
