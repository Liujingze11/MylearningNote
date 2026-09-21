import os
import subprocess
import threading
import time
from pathlib import Path

from huggingface_hub import hf_hub_download


MODEL = "sam3.pt"
BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = BASE_DIR / MODEL


# ==========================
# 方法1: huggingface_hub（官方推荐）
# ==========================

def download_with_hf_hub(repo, timeout=10):
    """用 huggingface_hub 下载，timeout 秒无响应则放弃"""

    print(f"[官方] 尝试 hf_hub_download: {repo}")

    result = None
    error = None

    def worker():
        nonlocal result, error
        try:
            result = hf_hub_download(
                repo_id=repo,
                filename=MODEL,
                local_dir=str(BASE_DIR),
                etag_timeout=timeout,
            )
        except Exception as e:
            error = e

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join(timeout=timeout + 5)  # 多等 5 秒给网络缓冲

    if thread.is_alive():
        print(f"[官方] {timeout}s 无响应，跳过")
        return None

    if result:
        print(f"[官方] 下载成功: {result}")
        return result

    print(f"[官方] 失败: {error}")
    return None


# ==========================
# 方法2: curl 直链下载（备用，实测可用）
# ==========================

def download_with_curl_mirror(repo="cubert-gmbh/sam3"):
    """从 hf-mirror 获取 302 重定向直链，再用 curl 下载"""

    mirror = "https://hf-mirror.com"
    resolve_url = f"{mirror}/{repo}/resolve/main/{MODEL}"

    print(f"[备用] 通过 hf-mirror 获取直链: {resolve_url}")

    # 1) 获取 302 重定向地址
    try:
        headers = subprocess.run(
            ["curl", "-sI", resolve_url],
            capture_output=True, text=True, timeout=15,
        ).stdout
    except subprocess.TimeoutExpired:
        print("[备用] 获取直链超时")
        return None

    redirect_url = None
    for line in headers.splitlines():
        if line.lower().startswith("location:"):
            redirect_url = line.split(":", 1)[1].strip()
            break

    if not redirect_url:
        print("[备用] 未获取到重定向地址")
        return None

    print(f"[备用] 开始下载（约 3.3GB，请耐心等待）...")

    # 2) 用 curl -L 直接下载
    proc = subprocess.run(
        ["curl", "-L", "--progress-bar",
         "-o", str(CHECKPOINT_PATH), redirect_url],
        timeout=600,
    )

    if proc.returncode == 0 and CHECKPOINT_PATH.exists():
        size_gb = CHECKPOINT_PATH.stat().st_size / (1024 ** 3)
        print(f"[备用] 下载成功: {CHECKPOINT_PATH} ({size_gb:.1f} GB)")
        return str(CHECKPOINT_PATH)

    print("[备用] 下载失败")
    return None


# ==========================
# 主流程
# ==========================

def main():
    if CHECKPOINT_PATH.exists():
        size_gb = CHECKPOINT_PATH.stat().st_size / (1024 ** 3)
        print(f"权重已存在: {CHECKPOINT_PATH} ({size_gb:.1f} GB)")
        return

    # 优先官方源
    sources_hf = [
        ("facebook/sam3", None),
        ("facebook/sam3", "https://hf-mirror.com"),
    ]

    for repo, endpoint in sources_hf:
        if endpoint:
            os.environ["HF_ENDPOINT"] = endpoint
            print(f"\n>>> 官方源（镜像: {endpoint}）")
        else:
            os.environ.pop("HF_ENDPOINT", None)
            print(f"\n>>> 官方源（直连）")

        path = download_with_hf_hub(repo)
        if path:
            return

    # 官方都失败，用备用方案
    print("\n>>> 备用方案: curl 直链下载")
    path = download_with_curl_mirror()
    if path:
        return

    print("\n所有下载方式均失败")


if __name__ == "__main__":
    main()
