function [beta, lambdaR] = getBestParams(noiseIdx)
    % noiseVar에 따른 최적 하이퍼파라미터 매핑 (그리드 서치 결과 반영)
    
    switch noiseIdx
        case 1
            beta = 0.99; lambdaR = 10;
        case 2
            beta = 0.98; lambdaR = 10;
        case 3
            beta = 0.99; lambdaR = 10;
        case 4
            beta = 0.99; lambdaR = 10;
        case 5
            beta = 0.98; lambdaR = 5.0;
        otherwise
            % 범위를 벗어나는 경우 기본값 설정
            beta = 0.90; lambdaR = 1.0; 
    end
end