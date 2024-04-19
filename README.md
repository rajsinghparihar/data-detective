# data-detective
An app that leverages LLMs to process documents, extract relevant information and provide a summary specific to financial data


# Setup

> Installing the correct version of llama-cpp-python \
[Resource](https://pypi.org/project/llama-cpp-python/)
- for NVIDIA GPU \
    ```CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python```
- for Apple silicon \
    ```CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python```
- for CPU \
    ```CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS" pip install llama-cpp-python```
> Installing requirements \
- `pip install -r requirements.txt`