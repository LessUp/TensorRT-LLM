#!/usr/bin/env python3
"""校验翻译后的 markdown 文档：外部链接是否与 git HEAD 中的原文一致。

用法:
    python3 scripts/check_translation_links.py <翻译后的文件路径> [更多路径...]

检测两类问题:
    1. 译文缺失原文中的链接（翻译时漏掉）
    2. 译文多出原文没有的链接（手误打错 URL）
"""
import re
import subprocess
import sys


def extract_urls(md_text: str) -> set:
    return set(re.findall(r'https?://[^\s\)\]]+', md_text))


def main() -> None:
    files = sys.argv[1:]
    if not files:
        print(__doc__)
        sys.exit(1)
    ok = True
    for path in files:
        old = subprocess.run(['git', 'show', f'HEAD:{path}'],
                             capture_output=True, text=True)
        if old.returncode != 0:
            print(f'{path}: 无法从 git HEAD 读取原文，跳过')
            continue
        urls_old = extract_urls(old.stdout)
        urls_new = extract_urls(open(path).read())
        missing = urls_old - urls_new
        extra = urls_new - urls_old
        if not missing and not extra:
            print(f'{path}: ✓ URL 全部一致（{len(urls_old)} 个）')
        else:
            ok = False
            print(f'{path}: ✗ URL 不一致')
            for u in sorted(missing):
                print(f'   缺失(原文有译文无): {u}')
            for u in sorted(extra):
                print(f'   多余/错误(译文有原文无): {u}')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
