import os
import folder_paths
import json

# 下载hg 模型到本地
def download_hg_model(model_id:str,exDir:str=''):
    # 下载本地
    model_checkpoint = os.path.join(folder_paths.models_dir, exDir, os.path.basename(model_id))
    print(model_checkpoint)
    if not os.path.exists(model_checkpoint):
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=model_id, local_dir=model_checkpoint, local_dir_use_symlinks=False)
    return model_checkpoint