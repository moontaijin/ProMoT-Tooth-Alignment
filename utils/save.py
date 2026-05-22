import os
import yaml
import shutil
from datetime import datetime

def save_config_snapshot(config_path: str, output_dir: str, cfg_dict: dict):
    """
    outputs/<exp>/ 에
      - config_used.yaml : 실제 사용된 YAML 원문 스냅샷
      - config_resolved.yaml : 파싱된 dict를 다시 dump(선택, 디버깅/재현용)
    를 저장한다.
    """
    os.makedirs(output_dir, exist_ok=True)

    config_path_abs = os.path.abspath(config_path)
    used_yaml_path = os.path.join(output_dir, "config_used.yaml")
    resolved_yaml_path = os.path.join(output_dir, "config_resolved.yaml")

    # 1) YAML 원문 그대로 저장 (가장 중요)
    with open(config_path_abs, "r", encoding="utf-8") as f:
        raw = f.read()
    with open(used_yaml_path, "w", encoding="utf-8") as f:
        f.write(raw)

    # 2) 파싱된 cfg를 다시 저장(키 순서 유지)
    with open(resolved_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_dict, f, sort_keys=False, allow_unicode=True)