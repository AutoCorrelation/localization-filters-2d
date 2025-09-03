# 코드 실행 방법  
- git clone 후 폴더 디렉토리에 `/data` 폴더 추가  
- src/`main.m` 파일 실행  
- 실행 시 do preSimulate? Y/N:  라고 묻는데 `/data` 내 시뮬레이션 데이터가 있으면 N, 초기 실행이라면 Y

 # parameter  
 |x|number of x|
 |--|--|
 |Particles|5e3|
 |Iteration(KF)|1e3|
 |Iteration(PF)|1e3|  

 # results with 1e4 iteration  
 |x|0.01|0.1|1|10|100|
 |--|--|-|-|-|-|
 |KF|0.0888|0.2832|0.8948|2.7468|8.172| 
 |KF1|0.0837|0.268|0.8567|2.6766|7.6151|
 |PF|0.0826|0.2654|0.8493|2.6594|7.4717|  
 
