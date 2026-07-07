import os
import re
import mne
import tempfile
import shutil
import numpy as np

def load_cnt_data(filepath, samplerate=256):
    """Loads a .CNT file via a temporary buffer and applies the standard 3D montage."""
    print(f"\n--- STEP 1: INGESTION ---")
    print(f"Processing: {filepath}")

    filename = os.path.basename(filepath)
    match = re.search(r'G(\d+)Day(\d+)([a-zA-Z0-9]+)\.cnt', filename, re.IGNORECASE)

    if match:
        print(f" -> Found: Subject {match.group(1)}, Day {match.group(2)}, Session {match.group(3)}")
    else:
        print(f" -> Warning: Could not parse metadata from filename: {filename}")

    # Load data into temporary local buffer
    temp_dir = tempfile.gettempdir()
    local_filepath = os.path.join(temp_dir, filename)
    raw = None

    try:
        print(f" -> Buffering file to local temporary storage...")
        shutil.copy2(filepath, local_filepath)
        print(f" -> Loading data into RAM...")
        try:
            raw = mne.io.read_raw_cnt(local_filepath, preload=True, data_format='int32')
        except RuntimeError:
            print(" -> 'int32' format failed, trying 'int16' format...")
            raw = mne.io.read_raw_cnt(local_filepath, preload=True, data_format='int16')
            
        aux_channels = ['HEOG', 'VEOG', 'EKG']
        present_aux = [ch for ch in aux_channels if ch in raw.ch_names]
        if present_aux:
            raw.drop_channels(present_aux)
            print(f" -> Dropped non-EEG channels: {present_aux}")

        # Downsample to target sample rate
        raw.filter(l_freq=1.0, h_freq=100.0, fir_design='firwin', verbose=False, phase='zero')
        raw.resample(256, npad='auto', window='hamming')

        # Crop the first and last 10 seconds to remove filter edge artifacts (ringing/transients)
        # which are huge due to DC offset and zero-phase padding
        raw.crop(tmin=10.0, tmax=raw.times[-1] - 10.0)

        # Apply average reference — ZUNA was trained on average-referenced data.
        # Skipping this presents data in a different spatial reference frame than
        # the training distribution, biasing reconstruction of frontal/occipital channels.
        raw.set_eeg_reference('average', projection=False, verbose=False)

        print(f" -> Resampled to 256Hz, cropped edge artifacts, average-referenced.")

    except Exception as e:
        print(f" -> Error during buffering/loading: {e}")
        return None
    finally:
        if os.path.exists(local_filepath):
            os.remove(local_filepath)
            print(" -> Temporary buffer cleared.")

    # Reclassify and apply standard 3D coordinates
    if raw is not None:
        try:
            chan_types = {}
            for ch in ['HEOG', 'VEOG']:
                if ch in raw.ch_names: chan_types[ch] = 'eog'
            if 'EKG' in raw.ch_names: chan_types['EKG'] = 'ecg'
            
            if chan_types:
                raw.set_channel_types(chan_types)
                print(f" -> Reclassified auxiliary channels: {list(chan_types.keys())}")

            montage = mne.channels.make_standard_montage('standard_1005')
            raw.set_montage(montage, match_case=False, on_missing='ignore')
            
            # THE FIX: Automatically find and drop any channels missing 3D coordinates
            nan_chs = [ch['ch_name'] for ch in raw.info['chs'] if np.isnan(ch['loc'][:3]).any()]
            if nan_chs:
                print(f" -> Dropping legacy/unmappable channels to prevent math errors: {nan_chs}")
                raw.drop_channels(nan_chs)

            print(" -> Success: Standard 3D spatial coordinates applied.")
        except Exception as e:
            print(f" -> Warning: Montage error: {e}")

    return raw

def create_epochs(raw, duration=5.0):
    """
    Creates overlapping 5-second epochs strictly locked to Neuroscan annotation markers.
    """
    print(f"\n--- STEP 2: EPOCHING (STRICT ANNOTATION WINDOW) ---")
    
    try:
        # 1. Extract events from digital Annotations rather than physical STIM channels
        events, event_dict = mne.events_from_annotations(raw)
        
        if len(events) > 0:
            # Diagnostic: Show how the text annotations map to integer codes
            print(f" -> Hardware diagnostic: Found these annotation mappings: {event_dict}")
            
            # ---------------------------------------------------------
            # RIGHT HERE: Enter your array of possible 0.5s trigger codes
            # ---------------------------------------------------------
            # Note: Look at the event_dict printout to confirm your code numbers!
            target_event_codes = [1, 2, 3, 4, 5, 6, 7, 8] 
            
            mask = np.isin(events[:, 2], target_event_codes)
            target_count = np.sum(mask)
            
            if target_count == 0:
                raise ValueError(f"CRITICAL: None of the target codes {target_event_codes} were found. Halting to preserve rigor.")

            print(f" -> Anchoring {target_count} epochs strictly to event codes {target_event_codes}...")
            
            # 2. Tell MNE to build epochs around ANY of the codes in our list
            # Calculate tmax to exclude the overlapping fencepost sample
            tmax_adjusted = duration - (1.0 / raw.info['sfreq'])
            epochs = mne.Epochs(raw, events, event_id=target_event_codes, tmin=0, tmax=tmax_adjusted, 
                                baseline=(None, None), preload=True, reject_by_annotation=False)
            
            print(f" -> Success: Created {len(epochs)} marker-locked overlapping epochs (with mean baseline correction).")
            return epochs
            
        else:
            raise ValueError("No annotations found in the file.")
            
    except Exception as e:
        print(f" -> EPOCHING FAILED: {e}")
        print(" -> Pipeline stopped to prevent arbitrary mathematical chopping.")
        return None

def degrade_channels(epochs, target_montage='19_channel'):
    """
    Degrades high-density data down to a standard clinical montage 
    (19-channel or 32-channel) to test spatial upscaling.
    """
    print(f"\n--- STEP 3: SPATIAL DEGRADATION ({target_montage.upper()}) ---")
    
    # Get all the current actual EEG channels in the file
    eeg_picks = mne.pick_types(epochs.info, eeg=True, exclude=[])
    all_channels = [epochs.ch_names[i].upper() for i in eeg_picks]
    
    # Define the exact electrodes to KEEP based on standard clinical layouts
    if target_montage == '19_channel':
        # The classic International 10-20 System
        keep_list = ['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 
                     'T7', 'C3', 'CZ', 'C4', 'T8', 
                     'P7', 'P3', 'PZ', 'P4', 'P8', 
                     'O1', 'O2']
                     
    elif target_montage == '32_channel':
        # Standard 32-channel layout (10-20 plus intermediate rows)
        keep_list = ['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 
                     'FC5', 'FC1', 'FC2', 'FC6', 
                     'T7', 'C3', 'CZ', 'C4', 'T8', 
                     'CP5', 'CP1', 'CP2', 'CP6', 
                     'P7', 'P3', 'PZ', 'P4', 'P8', 
                     'PO3', 'PO4', 'O1', 'O2', 'OZ']
    else:
        raise ValueError("target_montage must be '19_channel' or '32_channel'")

    # Figure out which channels to drop by subtracting the keep_list from all_channels
    dropped_channels = []
    kept_channels_found = []
    
    for ch in epochs.ch_names:
        if ch.upper() in keep_list:
            kept_channels_found.append(ch)
        elif ch.upper() in all_channels: # Only drop if it's actually an EEG channel
            dropped_channels.append(ch)

    # Apply the damage
    epochs.info['bads'] = dropped_channels
    
    print(f" -> Target: {target_montage}")
    print(f" -> Kept {len(kept_channels_found)} core channels.")
    print(f" -> Dropped {len(dropped_channels)} intermediate channels.")
    
    return epochs

def interpolate_baseline(epochs):
    """Uses traditional Spherical Spline Interpolation to reconstruct missing channels."""
    print(f"\n--- STEP 4: TRADITIONAL SPLINE RECONSTRUCTION ---")
    print(" -> Interpolating missing channels using 3D spherical splines...")
    
    spline_epochs = epochs.copy()
    spline_epochs.interpolate_bads(reset_bads=True)
    
    print(" -> Success: Missing channels mathematically reconstructed.")
    return spline_epochs

def export_zuna_tensors(pristine_epochs, degraded_epochs, spline_epochs, subject_id, session_name, out_dir, num_test_epochs=None):
    print(f"\n--- STEP 5: TENSOR EXTRACTION FOR ZUNA ---")
    
    # Force identical alphabetical channel sorting across all epoch objects
    sorted_names = sorted(pristine_epochs.ch_names)
    
    pristine_sorted = pristine_epochs.copy().reorder_channels(sorted_names)
    degraded_sorted = degraded_epochs.copy().reorder_channels(sorted_names)
    spline_sorted   = spline_epochs.copy().reorder_channels(sorted_names)

    # Now safely extract the aligned data matrices and convert Volts to Microvolts
    y_truth  = pristine_sorted.get_data(copy=True).astype(np.float32) * 1e6
    X_spline = spline_sorted.get_data(copy=True).astype(np.float32) * 1e6
    X_broken = degraded_sorted.get_data(copy=True).astype(np.float32) * 1e6
    
    # Physically zero out the bad channels in X_broken for the encoder simulation [cite: 274]
    bad_channel_names = degraded_sorted.info['bads']
    if bad_channel_names:
        bad_indices = [sorted_names.index(ch) for ch in bad_channel_names]
        X_broken[:, bad_indices, :] = 0.0
        print(f" -> SECURE: Zeroed out {len(bad_indices)} dropped channels in alphabetical alignment.")

    # Save arrays
    np.save(os.path.join(out_dir, f"{subject_id}_{session_name}_X_broken.npy"), X_broken)
    np.save(os.path.join(out_dir, f"{subject_id}_{session_name}_X_spline.npy"), X_spline)
    np.save(os.path.join(out_dir, f"{subject_id}_{session_name}_y_truth.npy"), y_truth)
    
    print(f" -> Success! Verified mathematical shape: {X_broken.shape}")
    
    # --- THE DIRECT ZUNA INJECTION BYPASS ---
    import torch
    pt_out_dir = "test_pt_out"
    os.makedirs(pt_out_dir, exist_ok=True)
    
    if num_test_epochs is not None:
        n_epochs = min(num_test_epochs, X_broken.shape[0]) 
    else:
        n_epochs = X_broken.shape[0] # Process all epochs
        
    X_inject = X_broken[:n_epochs]
    
    # --- Z-SCORE NORMALIZATION (matching ZUNA's native preprocessing) ---
    # ZUNA was trained on z-scored data. Its inference pipeline then divides by data_norm=10
    # to bring std from ~1.0 to ~0.1 (matching stft_global_sigma). Without z-scoring,
    # our raw µV data has std~16.5, which after /10 is still 1.65 — 16.5x too large,
    # causing 11% of the signal to be clipped at ±1.0.
    nonzero_mask = np.any(X_inject != 0.0, axis=-1)  # (epochs, channels) — True where channel is preserved
    nonzero_values = X_inject[nonzero_mask]           # all preserved samples flattened
    zscore_mean = float(nonzero_values.mean())
    zscore_std  = float(nonzero_values.std())
    
    X_inject_normed = X_inject.copy()
    # Normalize only preserved channels; dropped channels stay at 0.0
    for ep in range(X_inject_normed.shape[0]):
        for ch in range(X_inject_normed.shape[1]):
            if nonzero_mask[ep, ch]:
                X_inject_normed[ep, ch, :] = (X_inject_normed[ep, ch, :] - zscore_mean) / zscore_std
    
    print(f" -> Z-score normalization applied: mean={zscore_mean:.4f}, std={zscore_std:.4f}")
    print(f" -> Post-norm std: {X_inject_normed[nonzero_mask].std():.4f} (target: ~1.0)")
    print(f" -> After data_norm=10: std will be ~{X_inject_normed[nonzero_mask].std()/10:.4f} (target: 0.1)")
    
    # Extract native 3D coordinates from the montage for ZUNA
    pos_dict = degraded_sorted.get_montage().get_positions()['ch_pos']
    
    # --- CHANNEL BOUNDARY DIAGNOSTIC ---
    print("\n--- SURVIVING CHANNEL SPATIAL MANIFEST ---")
    for ch in degraded_sorted.ch_names:
        pos = pos_dict[ch]
        max_dist = np.max(np.abs(pos))
        if max_dist > 0.119:
            print(f" !!! OUTLIER: {ch:<4} | Max Extent: {max_dist:.4f}m | Coords: {pos}")
    print("------------------------------------------\n")
    
    pos_array = [pos_dict[ch] for ch in degraded_sorted.ch_names]
    pos_tensor = torch.tensor(np.array(pos_array), dtype=torch.float32)
    
    # THE FIX: Dynamically scale head model to fit ZUNA's strict [-0.12, 0.12] bounding box
    max_val = torch.max(torch.abs(pos_tensor))
    if max_val > 0.119:
        scale_factor = 0.119 / max_val
        pos_tensor = pos_tensor * scale_factor
        print(f" -> Scaled spatial coordinates by {scale_factor:.4f} to fit ZUNA bounding box.")
    
    # Build the exact dictionary ZUNA inference expects
    pt_dict = {
        'data': torch.tensor(X_inject_normed, dtype=torch.float32),
        'channel_positions': pos_tensor,
        'metadata': {
            'sfreq': 256,
            'ch_names': degraded_sorted.ch_names,
            'zscore_mean': zscore_mean,
            'zscore_std': zscore_std,
        }
    }
    
    # ZUNA strictly parses filenames to get tensor dimensions: prefix_epochs_channels_time.pt
    n_chans = X_inject.shape[1]
    n_time = X_inject.shape[2]
    
    # Camouflage the filename to match the ZUNA dataloader's exact expectations
    zuna_filename = f"ds000000_000000_000000_d00_{n_epochs:05d}_{n_chans}_{n_time}.pt"
    
    torch.save(pt_dict, os.path.join(pt_out_dir, zuna_filename))
    print(f" -> INJECTION SUCCESS: Saved as {zuna_filename}.")

# --- Execution Block ---
if __name__ == "__main__":
    # Load test file and parse metadata (relative to repo root; place raw .cnt under GEEG_Raw/)
    test_file = os.environ.get("GEEG_CNT", "GEEG_Raw/G001Day1Rest1.cnt")
    local_dir = os.path.dirname(os.path.abspath(__file__))

    filename = os.path.basename(test_file)
    match = re.search(r'G(\d+)Day(\d+)([a-zA-Z0-9]+)\.cnt', filename, re.IGNORECASE)
    subject_id = f"G{match.group(1)}" if match else "UnknownSubj"
    session_name = f"Day{match.group(2)}{match.group(3)}" if match else "UnknownSess"
    
    if os.path.exists(test_file):
        # Step 1. Ingest
        raw_data = load_cnt_data(test_file)
        
        if raw_data is not None:
            # Step 2. Chop
            epoched_data = create_epochs(raw_data, duration=5.0)

            if epoched_data is not None:
            
                # Step 3. Degrade to 19-channel
                degraded_data = degrade_channels(epoched_data, target_montage='19_channel')

                # Step 4. Reconstruct (Spline Baseline)
                reconstructed_data = interpolate_baseline(degraded_data)
                
                # Step 5. Export to ZUNA Tensors
                export_zuna_tensors(
                    pristine_epochs=epoched_data,
                    degraded_epochs=degraded_data,
                    spline_epochs=reconstructed_data,
                    subject_id=subject_id,
                    session_name=session_name,
                    out_dir=local_dir
                )

                print("\n*** PIPELINE COMPLETE! ***")
                
            else:
                print("\nPipeline stopped: Epoching failed.")
                
        else:
            print("\nPipeline stopped: Could not load data.")
    else:
        print(f"Error: Could not find {test_file}")