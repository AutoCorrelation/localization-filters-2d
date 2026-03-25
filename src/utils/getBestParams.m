function [beta, lambdaR] = getBestParams(noiseIdx)
    % Per-noise tuned hyperparameters (IMM baseline table).
    betaTable = [0.98, 0.99, 0.990, 0.90, 0.995];
    lambdaRTable = [10, 20, 200, 2, 20];

    if isempty(noiseIdx) || ~isfinite(noiseIdx)
        beta = 0.90;
        lambdaR = 1.0;
        return;
    end

    idx = max(1, min(round(noiseIdx), numel(betaTable)));
    beta = betaTable(idx);
    lambdaR = lambdaRTable(idx);
end