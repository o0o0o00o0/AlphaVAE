
import os
import shutil
from typing import List, Optional
from pathlib import Path
import fnmatch

def load_gitignore_patterns(gitignore_path: Path) -> List[str]:
    patterns = []
    if gitignore_path.exists():
        with gitignore_path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                patterns.append(line)
    return patterns

def is_ignored(path: str, patterns: List[str], root_dir: str) -> bool:
    rel_path = os.path.relpath(path, root_dir)
    for pattern in patterns:
        if fnmatch.fnmatch(rel_path, pattern) or fnmatch.fnmatch(os.path.basename(rel_path), pattern):
            return True
    return False

def copy_code_files(source_dir: str, target_dir: str, excludes: Optional[List[str]] = None):
    source_dir = os.path.abspath(source_dir)
    target_dir = os.path.abspath(target_dir)

    # 处理排除规则
    if excludes is not None:
        ignore_patterns = excludes
    else:
        gitignore_path = Path(source_dir) / ".gitignore"
        if gitignore_path.exists():
            ignore_patterns = load_gitignore_patterns(gitignore_path)
        else:
            ignore_patterns = ['output', 'outputs', 'save', 'saves', 'wandb', 'log', 'logs']

    for root, dirs, files in os.walk(source_dir):
        # 跳过符号链接目录
        dirs[:] = [d for d in dirs if not os.path.islink(os.path.join(root, d))]

        # 如果当前路径是 target_dir 或其子目录，跳过
        if os.path.commonpath([root, target_dir]) == target_dir:
            print(f"Skipping target_dir or its subdir: {root}")
            continue

        # 跳过被忽略的目录
        dirs[:] = [d for d in dirs if not is_ignored(os.path.join(root, d), ignore_patterns, source_dir)]

        for file in files:
            if not (file.endswith('.py') or file.endswith('.sh')):
                continue

            source_path = os.path.join(root, file)
            if is_ignored(source_path, ignore_patterns, source_dir):
                continue

            relative_path = os.path.relpath(source_path, source_dir)
            target_path = os.path.join(target_dir, relative_path)

            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            shutil.copy2(source_path, target_path)
            # print in cyan
            # print(f"\033[96mCopied: {source_path} -> {target_path}\033[0m")

# 示例用法：
# copy_code_files('/path/to/source', '/path/to/target')
# 或带排除规则：
# copy_code_files('/path/to/source', '/path/to/target', excludes=['*.ipynb', 'outputs/', 'wandb/'])
