% FMCW Radar Simulation (Ultrasonic - Doppler FFT, High Velocity Test)

%% 1. Define Parameters
f_start = 35e3;         % Start frequency (Hz)
f_end = 45e3;           % End frequency (Hz)
B = f_end - f_start;    % Bandwidth (Hz)

% *** PARAMETERS ADJUSTED FOR HIGH VELOCITY & HIGH PRF ***
Tc = 0.3e-3;        % Chirp Duration (seconds) - Very short (0.3 ms)
Fs = 1.5e6;         % Sampling Rate (Hz) - Increased significantly (1.5 MHz)
                    % Must be > 2 * max_beat_frequency.
                    % Max beat for R=3m, Tc=0.3ms is ~583 kHz. So Fs > 1.166 MHz.

c = 343;            % Speed of sound in air (m/s)
A = 1;              % Amplitude of transmitted signal

% Target Parameters
R = 3;             % Initial Target distance (meters)
v = 5;             % Target velocity (m/s) - positive away

attenuation_factor = 0.1;
Ar = A * attenuation_factor;

% Noise Parameters
SNR_dB = 20; % Keep SNR reasonable for clarity
SNR_linear = 10^(SNR_dB / 10);

% Multi-Chirp Parameters for Velocity
num_chirps = 64;     % Number of chirps for Doppler FFT
PRI = Tc + 0.1e-3;   % Pulse Repetition Interval = 0.3ms + 0.1ms = 0.4ms
PRF = 1/PRI;         % PRF = 1 / 0.0004 = 2500 Hz

%% Derived Parameters
lambda_center = c / (f_start + B/2);
time_axis_single_chirp = 0 : 1/Fs : Tc - 1/Fs; % Time vector for one chirp
N_samples = length(time_axis_single_chirp);
N_range_fft = 2^nextpow2(N_samples); % FFT size for range
f_axis_range = Fs/2 * linspace(0, 1, N_range_fft/2 + 1);
win_range = hann(N_samples)';

N_doppler_fft = 2^nextpow2(num_chirps);
f_axis_doppler = linspace(-PRF/2, PRF/2, N_doppler_fft);
win_doppler = hann(num_chirps)';

%% Initialize Storage
range_fft_matrix = zeros(N_range_fft, num_chirps);
complex_values_at_target_bin = zeros(1, num_chirps);

%% --- Simulation Loop for Multiple Chirps ---
target_range_bin = -1;
fb = NaN;
R_estimated = NaN;
Rx_plot = []; Mixed_Signal_Windowed_plot = []; FFT_magnitude_plot_positive_norm = []; max_magnitude_plot_val = NaN;

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
    FFT_range_result = fft(Mixed_Signal_Windowed, N_range_fft);
    range_fft_matrix(:, chirp_idx) = FFT_range_result;

    if chirp_idx == 1
        FFT_magnitude_abs = abs(FFT_range_result);
        FFT_magnitude_positive = FFT_magnitude_abs(1 : N_range_fft/2 + 1);
        
        % Heuristic for finding peak, considering potentially high beat frequency
        % Search in a plausible beat frequency range to avoid DC/noise if possible
        expected_fb_approx = (2 * R * B) / (c * Tc);
        search_range_hz = B * 0.5; % Search +/- half bandwidth around expected fb
        min_search_fb = max(0, expected_fb_approx - search_range_hz);
        max_search_fb = expected_fb_approx + search_range_hz;

        min_search_bin = max(2, floor(min_search_fb / (Fs/N_range_fft)));
        max_search_bin = min(N_range_fft/2+1, ceil(max_search_fb / (Fs/N_range_fft)));

        if min_search_bin >= max_search_bin
            % Fallback if search range is invalid
            [~, peak_idx_rel] = max(FFT_magnitude_positive(2:end)); % Ignore DC
            target_range_bin = peak_idx_rel + 1;
        else
            [~, peak_idx_rel_in_search] = max(FFT_magnitude_positive(min_search_bin:max_search_bin));
            target_range_bin = peak_idx_rel_in_search + min_search_bin - 1;
        end
        
        if isempty(target_range_bin) || target_range_bin <= 1 || target_range_bin > N_range_fft/2+1
            warning('Failed to find valid target peak in the first chirp Range FFT. Using fallback max.');
            [~, target_range_bin_fallback] = max(FFT_magnitude_positive(2:end));
             target_range_bin = target_range_bin_fallback + 1;
        end

        fb = f_axis_range(target_range_bin);
        R_estimated = (fb * c * Tc) / (2 * B);

        Rx_plot = Rx;
        Mixed_Signal_Windowed_plot = Mixed_Signal_Windowed;
        FFT_magnitude_plot_norm = abs(FFT_range_result / N_samples);
        FFT_magnitude_plot_positive_norm = FFT_magnitude_plot_norm(1:N_range_fft/2+1);
        FFT_magnitude_plot_positive_norm(2:end-1) = 2*FFT_magnitude_plot_positive_norm(2:end-1);
        max_magnitude_plot_val = max(FFT_magnitude_plot_positive_norm);
    end

    if target_range_bin > 0 && target_range_bin <= N_range_fft
        complex_values_at_target_bin(chirp_idx) = FFT_range_result(target_range_bin);
    else
         complex_values_at_target_bin(chirp_idx) = NaN;
    end
end

%% --- Doppler Processing ---
fd_estimated = NaN; v_estimated = NaN;
FFT_doppler_magnitude = NaN(1, N_doppler_fft);

if any(isnan(complex_values_at_target_bin))
    warning('NaN values found in complex data for Doppler FFT. Velocity calculation skipped.');
else
    complex_values_windowed = complex_values_at_target_bin .* win_doppler;
    FFT_doppler_result = fftshift(fft(complex_values_windowed, N_doppler_fft));
    FFT_doppler_magnitude = abs(FFT_doppler_result);
    [~, peak_idx_doppler] = max(FFT_doppler_magnitude);
    fd_estimated = f_axis_doppler(peak_idx_doppler);
    v_estimated = fd_estimated * lambda_center / 2;
end

%% --- Visualization ---
figure;
subplot(2, 1, 1);
if ~isempty(FFT_magnitude_plot_positive_norm) && ~isnan(fb)
    plot(f_axis_range / 1e3, FFT_magnitude_plot_positive_norm); hold on;
    plot(fb / 1e3, max_magnitude_plot_val, 'rv', 'MarkerSize', 8, 'LineWidth', 1.5); hold off;
    legend('FFT Magnitude', ['Peak (f_b = ', num2str(fb/1e3, '%.1f'), ' kHz)']); % Adjusted precision
else
    plot(f_axis_range / 1e3, ones(size(f_axis_range))*eps);
    legend('FFT Magnitude (Peak Detection Failed or Data Missing)');
end
title('Range FFT Spectrum (Chirp 1)');
xlabel('Beat Frequency (kHz)'); ylabel('Normalized Magnitude'); grid on;
xlim_max_range_fft = min(Fs/2e3, expected_fb_approx/1e3 + 2*search_range_hz/1e3); % Dynamic xlim
if isnan(xlim_max_range_fft) || xlim_max_range_fft <=0, xlim_max_range_fft = Fs/4e3; end
xlim([0 xlim_max_range_fft]);

subplot(2, 1, 2);
plot(f_axis_doppler, FFT_doppler_magnitude); hold on;
if ~isnan(fd_estimated)
    plot(fd_estimated, max(FFT_doppler_magnitude), 'rv', 'MarkerSize', 8, 'LineWidth', 1.5);
    legend('Doppler FFT Mag', ['Peak (f_d = ', num2str(fd_estimated, '%.2f'), ' Hz)']);
else
    legend('Doppler FFT Mag (Peak Detection Failed or Data Missing)');
end
hold off;
title('Doppler FFT Spectrum');
xlabel('Doppler Frequency (Hz)'); ylabel('Magnitude'); grid on;
xlim([-PRF/2 PRF/2]);

sgtitle(sprintf('FMCW High-V Test: R=%.1fm, v=%.1fm/s | Est R=%.2fm, Est v=%.3fm/s', ...
                 R, v, R_estimated, v_estimated));

if ~exist('Plots', 'dir'), mkdir('Plots'); end
saveas(gcf, 'Plots/FMCW_Simulation_DopplerFFT_HighV_Test.png');

%% --- Display Results ---
disp('--- Simulation Results (High Velocity Test) ---');
disp(['Target Range (Actual): ', num2str(R), ' m']);
disp(['Target Velocity (Actual): ', num2str(v), ' m/s']);
disp(['Chirp Duration (Tc): ', num2str(Tc*1e3), ' ms']);
disp(['Sampling Freq (Fs): ', num2str(Fs/1e3), ' kHz']);
disp(['Pulse Repetition Interval (PRI): ', num2str(PRI*1e3), ' ms']);
disp(['Pulse Repetition Frequency (PRF): ', num2str(PRF), ' Hz']);
disp(['Max Unambiguous Doppler: +/-', num2str(PRF/2), ' Hz']);
disp(['Max Unambiguous Velocity: +/-', num2str((PRF/2)*lambda_center/2), ' m/s']);
disp(['Expected Beat Frequency (approx): ', num2str((2 * R * B) / (c * Tc)), ' Hz']);
disp(['Detected Beat Frequency (fb, chirp 1): ', num2str(fb, '%.1f'), ' Hz']);
disp(['Estimated Range: ', num2str(R_estimated, '%.4f'), ' m']);
disp(['--- Velocity Calculation (Doppler FFT) ---']);
fd_true_high_v = 2*v/lambda_center;
disp(['Expected Doppler Frequency (fd_true = 2*v/lambda): ', num2str(fd_true_high_v, '%.2f'), ' Hz']);
disp(['Detected Doppler Frequency (fd_est): ', num2str(fd_estimated, '%.2f'), ' Hz']);
disp(['Estimated Velocity: ', num2str(v_estimated, '%.4f'), ' m/s']);