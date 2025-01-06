# DTLN.axera
Dual-signal Transformation LSTM Network on Axera

官方repo: (https://github.com/breizhn/DTLN)[https://github.com/breizhn/DTLN]

已提供转换好的axmodel: model_1.axmodel和model_2.axmodel  

## 运行

### ONNX
```
python run_ort.py -i noisy.wav -o clean_onnx.wav
```

### axmodel
```
python run_ax.py -i noisy.wav -o clean_ax.wav
```


如要自行转换一遍请参考如下步骤：

## 1. 安装依赖
```
pip install -r requirements.txt
```

## 2. 修改onnx模型
```
python modify_onnx.py
```
生成modified_model_1.onnx

## 3. 生成量化数据集
```
python generate_data.py
```
生成calibration_dataset

## 4. 转换成axmodel
model_1  
```
pulsar2 build --input modified_model_1.onnx --config config_model_1.json --output_dir model_1 --output_name model_1.axmodel --target_hardware AX650 --npu_mode NPU3
```
生成model_1/model_1.axmodel

model_2
```
pulsar2 build --input model_2.onnx --config config_model_2.json --output_dir model_2 -output_name model_2.axmodel --target_hardware AX650 --npu_mode NPU3
```
生成model_2/model_2.axmodel