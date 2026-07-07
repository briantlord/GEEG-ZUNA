import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import glob
import torch
import mne
from mne_connectivity import spectral_connectivity_epochs
from scipy.stats import ttest_rel, wilcoxon, pearsonr
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs
from mne.preprocessing import ICA

SUBJECT_ID   = "G001"
SESSION_NAME = "Day1Rest1"
SFREQ        = 256
NUM_EPOCHS   = 16
OUT_DIR      = "comparison_results"

BANDS = {
    'Alpha': (8, 13),
    'Beta': (13, 30),
}

def load_data():
    print("Loading data...")
    truth  = np.load(f"{SUBJECT_ID}_{SESSION_NAME}_y_truth.npy")[:NUM_EPOCHS]
    spline = np.load(f"{SUBJECT_ID}_{SESSION_NAME}_X_spline.npy")[:NUM_EPOCHS]
    zuna   = np.load(f"{SUBJECT_ID}_{SESSION_NAME}_X_zuna_test.npy")[:NUM_EPOCHS]

    pt_file = sorted(glob.glob("test_pt_out/*.pt"))[0]
    meta = torch.load(pt_file, weights_only=False).get('metadata', {})
    ch_names = meta.get('ch_names', [f"Ch{i}" for i in range(truth.shape[1])])
    ch_names_sorted = sorted(ch_names)

    # Convert uV to Volts for MNE
    truth = truth * 1e-6
    spline = spline * 1e-6
    zuna = zuna * 1e-6

    return truth, spline, zuna, ch_names_sorted

def create_epochs_objects(truth, spline, zuna, ch_names):
    info = mne.create_info(ch_names=ch_names, sfreq=SFREQ, ch_types='eeg')
    montage = mne.channels.make_standard_montage('standard_1005')
    info.set_montage(montage, match_case=False, on_missing='ignore')

    epochs_t = mne.EpochsArray(truth, info)
    epochs_s = mne.EpochsArray(spline, info)
    epochs_z = mne.EpochsArray(zuna, info)
    
    # Required for LORETA
    epochs_t.set_eeg_reference('average', projection=True)
    epochs_s.set_eeg_reference('average', projection=True)
    epochs_z.set_eeg_reference('average', projection=True)
    
    epochs_t.apply_proj()
    epochs_s.apply_proj()
    epochs_z.apply_proj()

    return epochs_t, epochs_s, epochs_z

# ── 1. Phase-Sensitive Connectivity ──────────────────────────────────────────
def test_connectivity(epochs_t, epochs_s, epochs_z):
    print("\n--- 1. Phase-Sensitive Connectivity (wPLI and PLV) ---")
    methods = ['wpli', 'plv']
    
    for method in methods:
        print(f"  Computing {method.upper()}...")
        
        # We compute connectivity for all bands at once
        fmin = [BANDS['Alpha'][0], BANDS['Beta'][0]]
        fmax = [BANDS['Alpha'][1], BANDS['Beta'][1]]
        
        con_t = spectral_connectivity_epochs(epochs_t, method=method, fmin=fmin, fmax=fmax, faverage=True, verbose=False)
        con_s = spectral_connectivity_epochs(epochs_s, method=method, fmin=fmin, fmax=fmax, faverage=True, verbose=False)
        con_z = spectral_connectivity_epochs(epochs_z, method=method, fmin=fmin, fmax=fmax, faverage=True, verbose=False)
        
        # output='dense' gives (n_nodes, n_nodes, n_bands)
        mat_t = con_t.get_data(output='dense')
        mat_s = con_s.get_data(output='dense')
        mat_z = con_z.get_data(output='dense')
        
        n_nodes = mat_t.shape[0]
        # MNE populates the lower triangle
        il = np.tril_indices(n_nodes, k=-1)
        
        for b_idx, b_name in enumerate(['Alpha', 'Beta']):
            err_s = np.abs(mat_t[:, :, b_idx] - mat_s[:, :, b_idx])[il]
            err_z = np.abs(mat_t[:, :, b_idx] - mat_z[:, :, b_idx])[il]
            
            # Wilcoxon signed-rank test on the absolute errors of the edges
            stat, pval = wilcoxon(err_s, err_z)
            
            mean_err_s = np.mean(err_s)
            mean_err_z = np.mean(err_z)
            
            winner = "Spline" if mean_err_s < mean_err_z else "ZUNA"
            
            print(f"    [{b_name}] Mean Edge Error -> Spline: {mean_err_s:.4f} | ZUNA: {mean_err_z:.4f} | Winner: {winner} | p = {pval:.2e}")
            
            # Plot Heatmaps
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            vmax = np.max(mat_t[:, :, b_idx])
            im0 = axes[0].imshow(mat_t[:, :, b_idx], vmin=0, vmax=vmax, cmap='viridis')
            axes[0].set_title(f'Truth ({method.upper()})')
            axes[1].imshow(mat_s[:, :, b_idx], vmin=0, vmax=vmax, cmap='viridis')
            axes[1].set_title(f'Spline ({method.upper()})')
            axes[2].imshow(mat_z[:, :, b_idx], vmin=0, vmax=vmax, cmap='viridis')
            axes[2].set_title(f'ZUNA ({method.upper()})')
            
            for ax in axes:
                ax.set_xticks([])
                ax.set_yticks([])
            
            cbar = fig.colorbar(im0, ax=axes.ravel().tolist(), shrink=0.8)
            cbar.set_label(f'{method.upper()}')
            plt.suptitle(f'Connectivity ({b_name} Band)\np={pval:.2e} (Spline vs ZUNA Error)')
            plt.savefig(os.path.join(OUT_DIR, f'adv_conn_{method}_{b_name}.png'), dpi=200)
            plt.close()

# ── 2. Source Estimation (LORETA) ────────────────────────────────────────────
def test_source_estimation(epochs_t, epochs_s, epochs_z):
    print("\n--- 2. Source Estimation (eLORETA) ---")
    fs_dir = mne.datasets.fetch_fsaverage(verbose=False)
    subjects_dir = os.path.dirname(fs_dir)
    subject = 'fsaverage'
    
    print("  Setting up source space and BEM...")
    src = mne.setup_source_space(subject, spacing='oct6', add_dist='patch', subjects_dir=subjects_dir, verbose=False)
    model = mne.make_bem_model(subject=subject, ico=4, subjects_dir=subjects_dir, conductivity=(0.3, 0.006, 0.3), verbose=False)
    bem = mne.make_bem_solution(model, verbose=False)
    
    print("  Computing forward solution...")
    fwd = mne.make_forward_solution(epochs_t.info, trans='fsaverage', src=src, bem=bem, eeg=True, meg=False, verbose=False)
    
    print("  Computing data covariance...")
    cov = mne.compute_covariance(epochs_t, method='empirical', verbose=False)
    
    print("  Making inverse operator...")
    inv = make_inverse_operator(epochs_t.info, fwd, cov, loose=0.2, depth=0.8, verbose=False)
    
    snr = 3.0
    lambda2 = 1.0 / snr ** 2
    
    print("  Applying dSPM to all epochs...")
    stcs_t = apply_inverse_epochs(epochs_t, inv, lambda2, method='dSPM', return_generator=False, verbose=False)
    stcs_s = apply_inverse_epochs(epochs_s, inv, lambda2, method='dSPM', return_generator=False, verbose=False)
    stcs_z = apply_inverse_epochs(epochs_z, inv, lambda2, method='dSPM', return_generator=False, verbose=False)
    
    # We will compute power in Alpha and Beta in specific ROIs
    labels = mne.read_labels_from_annot(subject, parc='aparc', subjects_dir=subjects_dir, verbose=False)
    # Let's pick a frontal and an occipital ROI
    rois = {
        'Frontal': [l for l in labels if 'superiorfrontal' in l.name],
        'Occipital': [l for l in labels if 'lateraloccipital' in l.name]
    }
    
    sfreq = SFREQ
    
    # Visual plot of the average epoch (evoked) source power
    evoked_t = epochs_t.average()
    evoked_s = epochs_s.average()
    evoked_z = epochs_z.average()
    
    stc_ev_t = mne.minimum_norm.apply_inverse(evoked_t, inv, lambda2, method='dSPM', verbose=False)
    stc_ev_s = mne.minimum_norm.apply_inverse(evoked_s, inv, lambda2, method='dSPM', verbose=False)
    stc_ev_z = mne.minimum_norm.apply_inverse(evoked_z, inv, lambda2, method='dSPM', verbose=False)
    
    # Save evoked stc plots as images using brain
    # NOTE: matplotlib 3D plotting for brains can be tricky headless, we will extract the data and plot flat heatmaps 
    # of the source vertices instead, to guarantee it works in an automated script.
    
    print("  Extracting ROI time courses & Statistical Testing...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    roi_plot_idx = 0
    for roi_name, roi_labels in rois.items():
        if not roi_labels: continue
        label = roi_labels[0] + roi_labels[1] if len(roi_labels)>1 else roi_labels[0]
        
                # Extract for each epoch
        for b_idx, (b_name, (fmin, fmax)) in enumerate(BANDS.items()):
            power_t = []
            power_s = []
            power_z = []
            
            for i in range(NUM_EPOCHS):
                tc_t = stcs_t[i].in_label(label).data.mean(axis=0) # average across vertices in ROI
                tc_s = stcs_s[i].in_label(label).data.mean(axis=0)
                tc_z = stcs_z[i].in_label(label).data.mean(axis=0)
                
                # Compute band power
                psd_t, f_t = mne.time_frequency.psd_array_welch(tc_t, sfreq=sfreq, fmin=fmin, fmax=fmax, n_fft=sfreq*2, verbose=False)
                psd_s, f_s = mne.time_frequency.psd_array_welch(tc_s, sfreq=sfreq, fmin=fmin, fmax=fmax, n_fft=sfreq*2, verbose=False)
                psd_z, f_z = mne.time_frequency.psd_array_welch(tc_z, sfreq=sfreq, fmin=fmin, fmax=fmax, n_fft=sfreq*2, verbose=False)
                
                power_t.append(np.mean(psd_t))
                power_s.append(np.mean(psd_s))
                power_z.append(np.mean(psd_z))
                
            err_s = np.abs(np.array(power_t) - np.array(power_s))
            err_z = np.abs(np.array(power_t) - np.array(power_z))
            
            stat, pval = ttest_rel(err_s, err_z)
            mean_err_s = np.mean(err_s)
            mean_err_z = np.mean(err_z)
            winner = "Spline" if mean_err_s < mean_err_z else "ZUNA"
            
            print(f"    [{roi_name} | {b_name}] Mean Error -> Spline: {mean_err_s:.2e} | ZUNA: {mean_err_z:.2e} | Winner: {winner} | p = {pval:.2e}")
            
            ax = axes[roi_plot_idx, b_idx]
            ax.bar(['Spline', 'ZUNA'], [mean_err_s, mean_err_z], color=['blue', 'red'], alpha=0.7)
            ax.set_title(f"{roi_name} ROI - {b_name} Band Error\np={pval:.2e}")
            ax.set_ylabel("Absolute Source Power Error")
        roi_plot_idx += 1
            
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'adv_source_roi_errors.png'), dpi=200)
    plt.close()

# ── 3. Independent Component Analysis (ICA) ──────────────────────────────────
def test_ica(epochs_t, epochs_s, epochs_z):
    print("\n--- 3. Independent Component Analysis (ICA) ---")
    n_components = 15
    print(f"  Fitting ICA ({n_components} components)...")
    
    ica_t = ICA(n_components=n_components, method='fastica', random_state=42, max_iter='auto')
    ica_t.fit(epochs_t, verbose=False)
    
    ica_s = ICA(n_components=n_components, method='fastica', random_state=42, max_iter='auto')
    ica_s.fit(epochs_s, verbose=False)
    
    ica_z = ICA(n_components=n_components, method='fastica', random_state=42, max_iter='auto')
    ica_z.fit(epochs_z, verbose=False)
    
    comp_t = ica_t.get_components() # (n_channels, n_components)
    comp_s = ica_s.get_components()
    comp_z = ica_z.get_components()
    
    # Statistical Testing: Spatial Correlation of Topographies
    corr_s = []
    corr_z = []
    
    for i in range(n_components):
        topo_t = comp_t[:, i]
        
        # Find best match in Spline
        best_s_corr = 0
        for j in range(n_components):
            r, _ = pearsonr(topo_t, comp_s[:, j])
            if abs(r) > abs(best_s_corr):
                best_s_corr = r
        corr_s.append(abs(best_s_corr))
        
        # Find best match in ZUNA
        best_z_corr = 0
        for j in range(n_components):
            r, _ = pearsonr(topo_t, comp_z[:, j])
            if abs(r) > abs(best_z_corr):
                best_z_corr = r
        corr_z.append(abs(best_z_corr))
        
    stat, pval = ttest_rel(corr_s, corr_z)
    mean_corr_s = np.mean(corr_s)
    mean_corr_z = np.mean(corr_z)
    winner = "Spline" if mean_corr_s > mean_corr_z else "ZUNA"  # For correlation, higher is better
    
    print(f"    Mean Spatial Correlation with Truth Topologies -> Spline: {mean_corr_s:.4f} | ZUNA: {mean_corr_z:.4f} | Winner: {winner} | p = {pval:.2e}")
    
    # Plot top 5 component topographies
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    
    # Workaround: MNE ica.plot_components requires a head model which might be tricky in some headless environments, 
    # but we can try it. If it fails, we fall back to simple channel scatter.
    try:
        for i in range(5):
            mne.viz.plot_topomap(comp_t[:, i], epochs_t.info, axes=axes[0, i], show=False)
            axes[0, i].set_title(f'Truth IC {i+1}')
            
            # Find matching component index for plotting
            best_idx_s = np.argmax([abs(pearsonr(comp_t[:, i], comp_s[:, j])[0]) for j in range(n_components)])
            mne.viz.plot_topomap(comp_s[:, best_idx_s], epochs_s.info, axes=axes[1, i], show=False)
            axes[1, i].set_title(f'Spline Match (r={corr_s[i]:.2f})')
            
            best_idx_z = np.argmax([abs(pearsonr(comp_t[:, i], comp_z[:, j])[0]) for j in range(n_components)])
            mne.viz.plot_topomap(comp_z[:, best_idx_z], epochs_z.info, axes=axes[2, i], show=False)
            axes[2, i].set_title(f'ZUNA Match (r={corr_z[i]:.2f})')
            
        plt.suptitle(f"Top 5 ICA Components - Spatial Fidelity\np={pval:.2e} (Spline vs ZUNA Spatial Correlation)")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, 'adv_ica_topomaps.png'), dpi=200)
        plt.close()
    except Exception as e:
        print(f"    Failed to plot topomaps (often headless MNE issue): {e}")

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    truth, spline, zuna, ch_names = load_data()
    epochs_t, epochs_s, epochs_z = create_epochs_objects(truth, spline, zuna, ch_names)
    
    test_connectivity(epochs_t, epochs_s, epochs_z)
    test_ica(epochs_t, epochs_s, epochs_z)
    test_source_estimation(epochs_t, epochs_s, epochs_z)
    
    print("\n--- ADVANCED TESTS COMPLETE ---")
