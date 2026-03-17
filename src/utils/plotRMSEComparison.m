function plotRMSEComparison(noiseVariance, baseline_RMSE, lkf_RMSE, lpf_RMSE, lkf_decayQ_RMSE, nl_pf_RMSE, adaptivePF_RMSE)
% plotRMSEComparison - Plot RMSE results for all filters on a semilogx figure
% Usage:
%   plotRMSEComparison(noiseVariance, baseline_RMSE, lkf_RMSE, lpf_RMSE, lkf_decayQ_RMSE, nl_pf_RMSE, adaptivePF_RMSE)

figure;
h = semilogx(noiseVariance, baseline_RMSE, '-o', ...
              noiseVariance, lkf_RMSE, '-s', ...
              noiseVariance, lpf_RMSE, '-^', ...
              noiseVariance, lkf_decayQ_RMSE, '-d', ...
              noiseVariance, nl_pf_RMSE, '-x', ...
              noiseVariance, adaptivePF_RMSE, '-h');

legend({'Baseline', 'LinearKF', 'LinearPF', 'LinearKF\_DecayQ', 'NonLinearPF', 'AdaptivePF'}, 'Location', 'northwest');
xlabel('Noise Variance');
ylabel('RMSE');
title('RMSE Comparison by Noise Level');
grid on;
set(h, 'LineWidth', 1.5);
end
