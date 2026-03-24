# PyTorch 2.6+ Compatibility Fix

## Issue

When running evaluation with PyTorch 2.6+, you may encounter this error:

```
_pickle.UnpicklingError: Weights only load failed. This file can still be loaded, to do so you have two options...
WeightsUnpickler error: Unsupported global: GLOBAL numpy._core.multiarray.scalar was not an allowed global by default.
```

## Root Cause

PyTorch 2.6 changed the default value of the `weights_only` argument in `torch.load()` from `False` to `True` for security reasons. This prevents loading pickled objects that could execute arbitrary code.

However, the ViMoE-VQA checkpoint contains numpy scalars that aren't in the default safe globals list.

## Solution

The fix has been implemented in `src/core/vivqa_eval_cli.py`:

```python
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
```

This tells PyTorch to load the checkpoint without strict weight validation. Since this checkpoint is from the project source, it's trusted.

## What Changed

**Before:**
```python
checkpoint = torch.load(checkpoint_path, map_location=device)
```

**After:**
```python
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
```

## Verification

The fix is already applied in your installation. If you encounter the error, make sure you have the latest version of the code:

```bash
cd vqa-model-builder
git pull origin main
```

## Why This Approach

1. **Simplest Solution**: Setting `weights_only=False` is the most straightforward fix
2. **Trusted Source**: The checkpoint is from the project, not external sources
3. **Backward Compatible**: Works with PyTorch <2.6 and >=2.6
4. **No Dependencies**: Doesn't require additional packages

## Alternative Approach (Not Used)

If you prefer stricter security, you could use:

```python
import torch.serialization

with torch.serialization.safe_globals([numpy._core.multiarray.scalar]):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
```

However, this requires more complex setup and isn't necessary for trusted checkpoints.

## Environment Requirements

Make sure you're using PyTorch 2.6+:

```bash
python -c "import torch; print(torch.__version__)"
```

If you're using an older PyTorch version, the `weights_only` parameter is ignored (no error).

## Further Troubleshooting

If you still encounter issues:

1. **Update PyTorch**:
   ```bash
   pip install --upgrade torch
   ```

2. **Check Checkpoint Integrity**:
   ```bash
   python -c "import torch; checkpoint = torch.load('checkpoints/RichardNguyen_ViMoE-VQA/best_generative_model.pt', weights_only=False); print('✓ Checkpoint loads successfully')"
   ```

3. **Check CUDA** (if using GPU):
   ```bash
   python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
   ```

4. **Clear Cache** (sometimes helps):
   ```bash
   rm -rf ~/.cache/torch
   pip cache purge
   ```

## References

- [PyTorch 2.6 Release Notes](https://pytorch.org/blog/pytorch-2.6-release/)
- [torch.load Documentation](https://pytorch.org/docs/stable/generated/torch.load.html)
- [weights_only Parameter](https://pytorch.org/docs/stable/generated/torch.load.html#torch.load)

---

**Fixed**: March 24, 2026  
**File**: `src/core/vivqa_eval_cli.py`  
**Line**: 54
