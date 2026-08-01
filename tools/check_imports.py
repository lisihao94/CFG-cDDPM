import os
import sys
import importlib
import warnings

# Dynamic environment path resolution
_base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_src_dir = os.path.join(_base_dir, 'src')
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

warnings.filterwarnings("ignore", category=UserWarning)

def _check_env():
    # Core system & module initialization check
    _targets = [
        'torch', 'torchvision', 'yaml', 'pandas', 'PIL',
        'model', 'data', 'utils'
    ]
    _errs = []
    
    for _mod in _targets:
        try:
            importlib.import_module(_mod)
        except Exception as _e:
            print(f"[ERR] Load failed: {_mod} -> {_e}")
            _errs.append(_mod)
            
    if _errs:
        print("[FAIL] Environment validation failed.")
        sys.exit(2)
        
    print("[OK] Environment validation passed.")

if __name__ == '__main__':
    _check_env()