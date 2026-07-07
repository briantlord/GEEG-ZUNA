import torch
import glob
import numpy as np

print('Running inspection...')

pt_in_files = sorted(glob.glob('test_pt_out/*.pt'))
if not pt_in_files:
    print('No input pt files found')
    exit()
pt_in = pt_in_files[0]

pt_out_files = sorted(glob.glob('test_recon_pt/*.pt'))
if not pt_out_files:
    print('No output pt files found')
    exit()
pt_out = pt_out_files[0]

d_in = torch.load(pt_in, weights_only=False)
d_out = torch.load(pt_out, weights_only=False)

print('=== INPUT .pt file ===')
print(f'Keys: {list(d_in.keys())}')
print(f'data shape: {d_in["data"].shape}')
print(f'data dtype: {d_in["data"].dtype}')
print(f'channel_positions shape: {d_in["channel_positions"].shape}')
print(f'metadata: {d_in.get("metadata", "No metadata")}')

data_in = d_in['data'].numpy()  
print(f'data stats: mean={data_in.mean():.6f}, std={data_in.std():.6f}, min={data_in.min():.6f}, max={data_in.max():.6f}')

n_ep, n_ch, n_t = data_in.shape
zero_mask = np.all(data_in == 0, axis=-1)
print(f'Epochs: {n_ep}, Channels: {n_ch}, Timepoints: {n_t}')
print(f'Dropped (zero) channels per epoch: {zero_mask.sum(axis=1)}')

# ZUNA divides by 10.0 internally
print(f'After internal data_norm=10, std would be {data_in.std()/10.0:.6f} (target is ~0.1)')

print('\n=== OUTPUT .pt file ===')
print(f'Keys: {list(d_out.keys())}')

def inspect_data(key, d):
    if key in d:
        recon = d[key]
        print(f'{key} type: {type(recon)}')
        if isinstance(recon, list):
            print(f'num samples: {len(recon)}')
            for i, r in enumerate(recon[:3]):
                if r is not None:
                    arr = np.array(r)
                    print(f'  sample {i}: shape={arr.shape}, mean={arr.mean():.4f}, std={arr.std():.4f}')
                else:
                    print(f'  sample {i}: None')

inspect_data('data_reconstructed', d_out)
inspect_data('data', d_out)
