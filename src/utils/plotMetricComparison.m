function plotMetricComparison(noiseVariance, baseline_Metric, lkf_Metric, lpf_Metric, lkf_decayQ_Metric, nl_pf_Metric, adaptivePF_Metric)
% plotMetricComparison - Plot Metric results for all filters on a semilogx figure
% Usage:
%   plotMetricComparison(noiseVariance, baseline_Metric, lkf_Metric, lpf_Metric, lkf_decayQ_Metric, nl_pf_Metric, adaptivePF_Metric)

figure;
h = semilogx(noiseVariance, baseline_Metric, '-o', ...
              noiseVariance, lkf_Metric, '-s', ...
              noiseVariance, lpf_Metric, '-^', ...
              noiseVariance, lkf_decayQ_Metric, '-d', ...
              noiseVariance, nl_pf_Metric, '-x', ...
              noiseVariance, adaptivePF_Metric, '-h');

legend({'Baseline', 'LinearKF', 'LinearPF', 'LinearKF\_DecayQ', 'NonLinearPF', 'AdaptivePF'}, 'Location', 'northwest');
xlabel('Noise Variance');
ylabel('Metric');
title('Metric Comparison by Noise Level');
grid on;
set(h, 'LineWidth', 1.5);
end
