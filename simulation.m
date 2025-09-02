% FMCW Radar Simulation (Ultrasonic - Doppler FFT Optimizations)

%% 1. Define Parameters
f_start = 35e3;
f_end = 45e3;
B = f_end - f_start;
Tc = 0.3e-3; % Short chirp for high PRF
Fs = 1.5e6;  % High sampling rate for high beat frequency
c = 343;
A = 1;

R = 3;
v = 5;

attenuation_factor = 0.1;
Ar = A * attenuation_factor;

SNR_dB = 20; % Can increase to 30-40dB to test noise impact
SNR_linear = 10^(SNR_dB / 10);

% --- OPTIMIZATION: Increased num_chirps ---
num_chirps = 128; % Increased for better Doppler resolution
PRI = Tc + 0.1e-3; % 0.4ms
PRF = 1/PRI;       % 2500 Hz

%% Derived Parameters
lambda_center = c / (f_start + B/2);
time_axis_single_chirp = 0 : 1/Fs : Tc - 1/Fs;
N_samples = length(time_axis_single_chirp);
N_range_fft = 2^nextpow2(N_samples);
f_axis_range = Fs/2 * linspace(0, 1, N_range_fft/2 + 1);
win_range = hann(N_samples)';

% --- OPTIMIZATION: Zero-padding for Doppler FFT ---
N_doppler_fft_zeropad = 2^nextpow2(num_chirps * 4); % e.g., 4x zero padding
f_axis_doppler = linspace(-PRF/2, PRF/2, N_doppler_fft_zeropad);
win_doppler = hann(num_chirps)';

%% Initialize Storage
complex_values_for_doppler = zeros(1, num_chirps); % Renamed for clarity

%% --- Simulation Loop ---
fb_first_chirp = NaN;
R_estimated_first_chirp = NaN;
% For plotting the first chirp's range FFT
Rx_plot_first = []; Mixed_plot_first = []; FFT_mag_plot_first = []; max_mag_plot_first = NaN;


for chirp_idx = 1:num_chirps
    current_R = R + v * (chirp_idx - 1) * PRI;
    current_tau = 2 * current_R / c;

    Tx = A * sin(2 * pi * (f_start * time_axis_single_chirp + (B / (2 * Tc)) * time_axis_single_chirp.^2));

    Rx_ideal_extended = zeros(1, N_samples + ceil(current_tau*Fs) + 10);
    t_rx_extended = (0:length(Rx_ideal_extended)-1)/Fs;
    for i = 1:length(t_rx_extended)
        time_inst = t_rx_extended(i) - current_tau;
        if time_inst >= 0 && time_inst < Tc
             Rx_ideal_extended(i) = Ar * sin(2 * pi * (f_start * time_inst + (B / (2 * Tc)) * time_inst.^2));
        end
    end
    Rx_ideal = Rx_ideal_extended(1:N_samples);

    signal_power = mean(Rx_ideal.^2);
    noise_power = signal_power / SNR_linear;
    if noise_power <= 0 || ~isfinite(noise_power), noise_power = eps; end
    noise = sqrt(noise_power) * randn(size(Rx_ideal));
    Rx = Rx_ideal + noise;

    Mixed_Signal = Tx .* Rx;
    Mixed_Signal_Windowed = Mixed_Signal .* win_range;
    FFT_range_result_current_chirp = fft(Mixed_Signal_Windowed, N_range_fft);

    % --- OPTIMIZATION: Find peak FOR THIS CHIRP to address range migration ---
    FFT_mag_abs_current = abs(FFT_range_result_current_chirp);
    FFT_mag_pos_current = FFT_mag_abs_current(1 : N_range_fft/2 + 1);
    
    % Simple peak search (can be refined if needed, e.g. search around expected)
    [~, peak_idx_rel_current] = max(FFT_mag_pos_current(2:end)); % Ignore DC
    current_target_range_bin = peak_idx_rel_current + 1;

    if isempty(current_target_range_bin) || current_target_range_bin <=1 || current_target_range_bin > N_range_fft
        complex_values_for_doppler(chirp_idx) = NaN; % Mark as invalid
        if chirp_idx == 1, warning('Peak detection failed in range FFT for chirp 1.'); end
        continue; % Skip to next chirp if peak finding fails
    end
    
    complex_values_for_doppler(chirp_idx) = FFT_range_result_current_chirp(current_target_range_bin);

    % Store results from the first chirp for range estimation and plotting
    if chirp_idx == 1
        fb_first_chirp = f_axis_range(current_target_range_bin);
        R_estimated_first_chirp = (fb_first_chirp * c * Tc) / (2 * B);
        
        Rx_plot_first = Rx;
        Mixed_plot_first = Mixed_Signal_Windowed;
        FFT_mag_plot_first_norm = abs(FFT_range_result_current_chirp / N_samples);
        FFT_mag_plot_first = FFT_mag_plot_first_norm(1:N_range_fft/2+1);
        FFT_mag_plot_first(2:end-1) = 2*FFT_mag_plot_first(2:end-1);
        max_mag_plot_first = max(FFT_mag_plot_first);
    end
end

%% --- Doppler Processing ---
fd_estimated = NaN; v_estimated = NaN;
FFT_doppler_magnitude = NaN(1, N_doppler_fft_zeropad);

if any(isnan(complex_values_for_doppler))
    warning('NaN values in complex data for Doppler. Check range peak finding.');
else
    complex_values_windowed = complex_values_for_doppler .* win_doppler;
    % --- OPTIMIZATION: Using zero-padded FFT length ---
    FFT_doppler_result = fftshift(fft(complex_values_windowed, N_doppler_fft_zeropad));
    FFT_doppler_magnitude = abs(FFT_doppler_result);
    
    [peak_mag_doppler, peak_idx_doppler] = max(FFT_doppler_magnitude);

    % --- OPTIMIZATION: Parabolic Peak Interpolation for Doppler ---
    if peak_idx_doppler > 1 && peak_idx_doppler < N_doppler_fft_zeropad
        alpha = FFT_doppler_magnitude(peak_idx_doppler - 1);
        beta = FFT_doppler_magnitude(peak_idx_doppler); % Same as peak_mag_doppler
        gamma = FFT_doppler_magnitude(peak_idx_doppler + 1);
        % p is offset from peak_idx_doppler in bins, range -0.5 to 0.5
        p_offset = 0.5 * (alpha - gamma) / (alpha - 2*beta + gamma);
        interpolated_idx_doppler = peak_idx_doppler + p_offset;
        fd_estimated = f_axis_doppler(peak_idx_doppler) + p_offset * (PRF / N_doppler_fft_zeropad); % More precise
        % Alternative: fd_estimated = interp1(1:N_doppler_fft_zeropad, f_axis_doppler, interpolated_idx_doppler);
    else
        fd_estimated = f_axis_doppler(peak_idx_doppler); % Use bin center if at edges
        warning('Doppler peak at edge of spectrum, interpolation skipped.');
    end
    
    v_estimated = fd_estimated * lambda_center / 2;
end

%% --- Visualization ---
figure;
subplot(2, 1, 1);
if ~isempty(FFT_mag_plot_first) && ~isnan(fb_first_chirp)
    plot(f_axis_range / 1e3, FFT_mag_plot_first); hold on;
    plot(fb_first_chirp / 1e3, max_mag_plot_first, 'rv', 'MarkerSize', 8); hold off;
    legend('FFT Mag', ['Peak (f_b = ', num2str(fb_first_chirp/1e3, '%.1f'), ' kHz)']);
else
    plot(f_axis_range / 1e3, ones(size(f_axis_range))*eps);
    legend('FFT Mag (Peak Fail/No Data)');
end
title('Range FFT Spectrum (Chirp 1)');
xlabel('Beat Freq (kHz)'); ylabel('Norm Mag'); grid on;
expected_fb_approx_plot = (2 * R * B) / (c * Tc);
xlim_max_plot = min(Fs/2e3, expected_fb_approx_plot/1e3 * 1.5);
if isnan(xlim_max_plot) || xlim_max_plot <=0, xlim_max_plot = Fs/4e3; end
xlim([0 xlim_max_plot]);

subplot(2, 1, 2);
plot(f_axis_doppler, FFT_doppler_magnitude); hold on;
if ~isnan(fd_estimated)
    plot(fd_estimated, peak_mag_doppler, 'rv', 'MarkerSize', 8);
    legend('Doppler FFT', ['Interp. Peak (f_d = ', num2str(fd_estimated, '%.2f'), ' Hz)']);
else
    legend('Doppler FFT (Peak Fail/No Data)');
end
hold off;
title(['Doppler FFT Spectrum (', num2str(num_chirps), ' chirps, ZeroPad x', num2str(N_doppler_fft_zeropad/num_chirps) ')']);
xlabel('Doppler Freq (Hz)'); ylabel('Mag'); grid on; xlim([-PRF/2 PRF/2]);

sgtitle(sprintf('FMCW High-V Opt: R=%.1fm,v=%.1fm/s|Est R=%.2fm,Est v=%.3fm/s', ...
                 R, v, R_estimated_first_chirp, v_estimated));
if ~exist('Plots', 'dir'), mkdir('Plots'); end
saveas(gcf, 'Plots/FMCW_Simulation_DopplerFFT_HighV_Optimized.png');

%% --- Display Results ---
disp('--- Simulation Results (High Velocity Test - Optimized) ---');
disp(['Target Range (Actual): ', num2str(R), ' m']);
disp(['Target Velocity (Actual): ', num2str(v), ' m/s']);
disp(['Chirp Duration (Tc): ', num2str(Tc*1e3), ' ms, Samples/Chirp: ', num2str(N_samples)]);
disp(['Sampling Freq (Fs): ', num2str(Fs/1e3), ' kHz']);
disp(['PRF: ', num2str(PRF), ' Hz, Num Chirps: ', num2str(num_chirps)]);
disp(['Max Unambiguous Doppler: +/-', num2str(PRF/2), ' Hz']);
disp(['Max Unambiguous Velocity: +/-', num2str((PRF/2)*lambda_center/2, '%.2f'), ' m/s']);
disp(['Expected Beat Frequency (approx): ', num2str((2 * R * B) / (c * Tc), '%.1f'), ' Hz']);
disp(['Detected Beat Frequency (fb, chirp 1): ', num2str(fb_first_chirp, '%.1f'), ' Hz']);
disp(['Estimated Range (from chirp 1): ', num2str(R_estimated_first_chirp, '%.4f'), ' m']);
disp(['--- Velocity Calculation (Doppler FFT Optimized) ---']);
fd_true_high_v = 2*v/lambda_center;
disp(['Expected Doppler Frequency (fd_true = 2*v/lambda): ', num2str(fd_true_high_v, '%.2f'), ' Hz']);
disp(['Detected Interpolated Doppler Frequency (fd_est): ', num2str(fd_estimated, '%.2f'), ' Hz']);
disp(['Estimated Velocity: ', num2str(v_estimated, '%.4f'), ' m/s']);