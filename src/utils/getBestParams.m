function [beta, lambdaR] = getBestParams(noiseIdx)
    % Per-noise tuned hyperparameters (IMM baseline table).
    betaTable = [0.999, 0.995, 0.990, 0.999, 0.995];
    lambdaRTable = [0.01, 0.1, 2, 50, 0.5];

    if isempty(noiseIdx) || ~isfinite(noiseIdx)
        beta = 0.90;
        lambdaR = 1.0;
        return;
    end

    idx = max(1, min(round(noiseIdx), numel(betaTable)));
    beta = betaTable(idx);
    lambdaR = lambdaRTable(idx);
end