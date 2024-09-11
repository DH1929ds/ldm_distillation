# 기본 이미지 설정 (Miniconda 사용)
FROM continuumio/miniconda3:latest

# 작업 디렉토리 설정
WORKDIR /workspace

# 기본적인 설치 (필수적인 패키지 설치)
RUN conda install -y python=3.8

# 컨테이너 실행 시 셸을 기본으로 실행하도록 설정
CMD ["/bin/bash"]
