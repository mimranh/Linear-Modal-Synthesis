%% Rayleigh Damping Analysis: Comprehensive Wood Study
clc; clear; close all;

% 1. Load the Data
fname = 'wood_modal_data.json';
if ~exist(fname, 'file')
    error('File %s not found. Please ensure Python has saved the log.', fname);
end
val = jsondecode(fileread(fname));

% Extract Metadata
alpha = val.metadata.alpha;
beta = val.metadata.beta;
fs = val.metadata.fs;

% Extract All Modes from Python Log
f_all = [val.all_modes.f]';
d_all = [val.all_modes.d]';
A_all = [val.all_modes.A]';

% 2. Define MATLAB-side Filtering (Mirroring your Python logic)
% We identify 'clean' modes vs 'outliers' for visualization
freq_limit = 2000; % Hz
is_clean = (f_all < freq_limit);

f_clean = f_all(is_clean);
d_clean = d_all(is_clean);
A_clean = A_all(is_clean);

f_outlier = f_all(~is_clean);
d_outlier = d_all(~is_clean);

%% 3. Theoretical Rayleigh Curve Generation
% Generate a smooth curve based on the Alpha/Beta from Python
f_plot = linspace(10, max(f_all)*1.2, 2000);
omega_plot = 2 * pi * f_plot;
d_theory = 0.5 * (alpha ./ omega_plot + beta .* omega_plot);

%% 4. Plotting Results
figure('Color', 'w', 'Name', 'Modal Analysis: Wood vs. Rayleigh Model', 'Position', [100, 100, 1100, 500]);

% --- Subplot 1: Damping Space Analysis ---
subplot(1, 2, 1);
hold on; grid on; box on;

% Plot theory line
plot(f_plot, d_theory, 'Color', [0.2 0.2 0.2], 'LineWidth', 2.5, 'DisplayName', 'Theoretical Fit');

% Plot filtered (clean) modes
scatter(f_clean, d_clean, 60, 'filled', 'MarkerFaceColor', [0 0.447 0.741], ...
    'MarkerEdgeColor', 'k', 'DisplayName', 'Analyzed Wood Body');

% Plot outliers (The piercing frequencies)
if ~isempty(f_outlier)
    scatter(f_outlier, d_outlier, 60, 'r', 'x', 'LineWidth', 1.5, 'DisplayName', 'High-Freq Outliers');
end

xlabel('Frequency (Hz)', 'FontSize', 12);
ylabel('Damping Parameter (d)', 'FontSize', 12);
title(['Rayleigh Fit: \alpha = ', num2str(alpha, '%.2f'), ', \beta = ', num2str(beta, '%.6f')], 'FontSize', 14);
legend('Location', 'best');
xlim([0, max(f_all)*1.1]);
ylim([0, max(d_all)*1.2]);

% --- Subplot 2: Decay Time (T60) Analysis ---
% T60 is the time it takes for a mode to decay by 60dB
% Formula: T60 = 3 * ln(10) / d  approx 6.91 / d
subplot(1, 2, 2);
hold on; grid on; box on;

T60_actual = 6.91 ./ d_clean;
T60_theory = 6.91 ./ (0.5 * (alpha./(2*pi*f_clean) + beta*(2*pi*f_clean)));

% Error bars showing the gap between real wood and the mathematical model
for i = 1:length(f_clean)
    line([f_clean(i) f_clean(i)], [T60_theory(i) T60_actual(i)], 'Color', [0.8 0.8 0.8], 'HandleVisibility', 'off');
end

scatter(f_clean, T60_actual, A_clean*2, 'DisplayName', 'Measured T60');
plot(f_clean, T60_theory, 'k+', 'MarkerSize', 1, 'DisplayName', 'Synthesized T60');

xlabel('Frequency (Hz)', 'FontSize', 12);
ylabel('Decay Time T_{60} (seconds)', 'FontSize', 12);
title('Decay Discrepancy (Time Domain Impact)', 'FontSize', 14);
legend('Location', 'best');

% Print Summary to Console
fprintf('\n--- MATLAB Analysis Summary ---\n');
fprintf('Sample Rate: %d Hz\n', fs);
fprintf('Total Modes Logged: %d\n', length(f_all));
fprintf('Modes in Body (Used): %d\n', sum(is_clean));
fprintf('Mean Damping in Wood Body: %.2f\n', mean(d_clean));