# LeNet5-with-OpenFHE

```
LENET5-WITH-OPENFHE
├── build/
├── pytorch_LeNet5/
│   ├── __pycache__/
│   ├── coeffResult/
│   ├── degreeResult/
│   ├── parameters_standard/
│   ├── __init__.py
│   ├── inference_only.py
│   ├── utils_approx.py
├── results/
├── src/
│   ├── conv_bn_module.cpp
│   ├── conv_bn_module.h
│   ├── fc_layer.cpp
│   ├── fc_layer.h
│   ├── main.cpp
│   ├── relu.cpp
│   ├── relu.h   
├── CMakeLists.txt
├── autotest.sh
├── README.md
├── main.py
├── test_fhe.py
```

1. Docker 및 Visual Studio Code 설치
Docker 설치

## Windows: Docker Desktop for Windows 설치

    필수 구성: WSL2 활성화(권장) 또는 Hyper-V

    확인: PowerShell에서

    wsl --version


## macOS(Apple Silicon/Intel): Docker Desktop for Mac 설치

## Linux (Ubuntu 예시):

    sudo apt-get update
    sudo apt-get install -y ca-certificates curl gnupg
    sudo install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
    https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" \
    | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    sudo usermod -aG docker $USER
    # 로그아웃/로그인 후 아래로 확인
    docker --version

# VS Code 설치 + 확장
## VS Code 설치
## 필수 확장:
- Docker (Microsoft)
- Dev Containers (Microsoft)

# Docker 실행
## Docker 데몬 기동
- Windows/macOS: Docker Desktop 실행 (트레이/메뉴바에서 고래 아이콘 확인)
- Linux:
```
sudo systemctl enable --now docker
systemctl status docker
```

# Docker 파일 다운로드
> 파일다운로드 링크: 
> 다운받은 압축파일을 C:\ 로 이동
> 압축해제
# Visual Studio 실행
## ctrl + ` 을 통해 Termnial 실행
## C:\로 이동
```
cd C:\
cd lenet5-fhe
```
# Docker build 수행
```
docker build -t openfhe-lenet5:latest .
```
# Docker 실행 (반드시 build 이후 수행)
```
docker run --rm -it openfhe-lenet5
```
# Openfhe 설치
> cd openfhe-configurator
## Openfhe-Hexl 설치
> ./scripts/configure.sh
> n
> y
> ./scripts/build-openfhe-development.sh

# python code 실행
> conda activate py_3_10
> cd .. && cd LeNet5-with-Openfhe
> python main.py


# ReLU 변경 및 적용 방법
1. cd ~/LeNet5-with-OpenFHE/src
2. vim relu.cpp
3. relu.cpp 파일 내  ApproxReLU4_Student 하단 영역 수정
```
    // Insert your own approximation below

    auto result = ct_x; // modify this when implementing your own code

    // Insert your own approximation above
```
4. 적용 확인 시 6 선택
